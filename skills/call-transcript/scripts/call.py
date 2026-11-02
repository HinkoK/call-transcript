#!/usr/bin/env python3
"""Транскрипт записи созвона через Groq Whisper.

Подкоманды идут в порядке работы:
  prepare     звук из записи, паузы, куски по ~10 минут с разрезом на паузе
  langs       какой язык Whisper сам слышит в окнах по всей записи
  transcribe  прогон кусков с заданным языком: сырой JSON и пронумерованные сегменты
  check       пустые места, дыры, выдумки на тишине, повторы, съехавшее время
  compare     основной и справочный прогоны рядом по времени
  clip        короткие отрывки заново, на одном или нескольких языках
  assemble    правки из edits.json, говорящие, абзацы, итоговый .md

Быстрый режим:
  soniox      звук в Soniox: текст и говорящие по голосу, segments_soniox; у Soniox всё удаляется
  names       имена говорящим Soniox («1=Я 2=Олег»): turns в edits.json

Ключи и настройки берутся из окружения или из ~/.config/call-transcript/.env
(GROQ_API_KEY, SONIOX_API_KEY, CALL_TRANSCRIPT_*) и нигде не печатаются.
Готовые ответы Groq лежат в рабочей папке и повторно не запрашиваются.

Запасной движок: если Groq не дал ответа (лимит, сеть, ошибка, нет ключа), кусок
распознаётся локально через whisper.cpp и large-v3, ответ приводится к виду Groq.
При лимите и сбоях в рабочей папке ставится пауза Groq (groq_pause.json), и следующие
команды до её конца сразу идут на локальный движок. CALL_TRANSCRIPT_ENGINE=groq - без
запасного движка, =local - только локально.
"""
from __future__ import annotations

import argparse
import http.client
import io
import json
import math
import os
import re
import shutil
import signal
import ssl
import subprocess
import tempfile
import time
import uuid
import zlib
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENV_FILE = Path.home() / ".config/call-transcript/.env"


def read_env_file() -> dict[str, str]:
    values = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.strip().partition("=")
            name = name.strip().removeprefix("export ").strip()
            if sep and name and not name.startswith("#"):
                values[name] = value.split(" #")[0].strip().strip("'\"")
    return values


FILE_ENV = read_env_file()


def setting(name: str, default: str = "") -> str:
    """Сначала окружение, потом ~/.config/call-transcript/.env."""
    return os.environ.get(name, "").strip() or FILE_ENV.get(name, "") or default


ENDPOINT = os.environ.get("CALL_TRANSCRIPT_GROQ_URL") or "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3"
# auto - Groq, при сбое локально; groq - только Groq; local - только локально.
ENGINE = setting("CALL_TRANSCRIPT_ENGINE", "auto")
WHISPER_CPP = setting("CALL_TRANSCRIPT_WHISPER_CPP", "whisper-cli")
LOCAL_MODEL = Path(setting("CALL_TRANSCRIPT_LOCAL_MODEL",
                           str(Path.home() / ".cache/call-transcript/ggml-large-v3-q5_0.bin"))).expanduser()
# Подсказка с пунктуацией: без неё whisper.cpp пишет строчными и без знаков. Только при
# заданном языке - при auto подсказка тянула бы определение языка в свою сторону.
LOCAL_PROMPTS = {
    "ru": "Привет! Да, конечно. Давай созвонимся, обсудим.",
    "uk": "Привіт! Так, звісно. Давай зідзвонимося, обговоримо.",
    "en": "Hi! Yes, of course. Let's talk it over.",
}
# Groq называет язык по-английски полностью, whisper.cpp - кодом.
LANG_NAMES = {"ru": "Russian", "uk": "Ukrainian", "en": "English", "be": "Belarusian", "pl": "Polish",
              "de": "German", "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese"}
GROQ_PAUSE_FILE = "groq_pause.json"
SONIOX_API = os.environ.get("CALL_TRANSCRIPT_SONIOX_URL") or "https://api.soniox.com/v1"
# Подмена адреса - только для проверок: ключ уходит в заголовке, поэтому не на чужой http.
if not SONIOX_API.startswith(("https://", "http://127.0.0.1")):
    raise SystemExit("CALL_TRANSCRIPT_SONIOX_URL: только https:// или http://127.0.0.1")
SONIOX_MODEL = "stt-async-v5"
# Подсказка словами: имена и жаргон, которые Whisper путал (references/mishearings.md).
# Свои слова навсегда - в CALL_TRANSCRIPT_TERMS через запятую, на одну запись - --terms.
SONIOX_TERMS = ["Claude", "Claude Code", "CLAUDE.md", "Anthropic", "ChatGPT", "GPT", "AI-шка", "вайб-кодинг",
                "нейронка", "Telegram"]
SONIOX_WAIT = 1800  # секунд на распознавание; час звука у Soniox - около 2,5 минуты
CHUNK_SECONDS = 600   # кусок около 10 минут, разрез ищем на паузе в пределах полутора минут
LONG_PAUSE = 6.0      # тишина от 6 секунд считается настоящей паузой, речи там нет
# Фразы, которые Whisper выдумывает на тишине и в конце записи.
HALLUCINATIONS = (
    "дякую за перегляд", "дякуємо за перегляд", "спасибо за просмотр", "продолжение следует",
    "субтитры сделал", "субтитры создавал", "редактор субтитров", "корректор", "подписывайтесь",
    "ставьте лайк", "thank you for watching", "thanks for watching", "amara.org",
)


def stamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def parse_time(spec: str) -> float:
    seconds = 0.0
    for part in spec.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def load_key() -> str:
    key = setting("GROQ_API_KEY")
    if not key and (ENGINE == "groq" or (ENGINE == "auto" and local_problem())):
        raise SystemExit(f"нет GROQ_API_KEY ни в окружении, ни в {ENV_FILE}. "
                         "Ключ бесплатный: console.groq.com/keys")
    # Без ключа в режиме auto распознаёт локальный движок, recognize() говорит об этом в выводе.
    return key


class GroqFailed(Exception):
    """Groq не дал ответа. pause - сколько секунд его потом не трогать, 0 - не ставить паузу."""

    def __init__(self, reason: str, pause: float = 0.0) -> None:
        super().__init__(reason)
        self.reason = reason
        self.pause = pause


def whisper(key: str, path: Path, language: str | None) -> dict:
    """Один запрос к Groq. Ответ возвращается целиком, даже с пустыми сегментами."""
    fields = {"model": MODEL, "response_format": "verbose_json", "temperature": "0"}
    if language:
        fields["language"] = language
    boundary = f"----call{uuid.uuid4().hex}"
    buf = io.BytesIO()
    for name, value in fields.items():
        buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
              f"Content-Type: audio/mpeg\r\n\r\n".encode())
    buf.write(path.read_bytes())
    buf.write(f"\r\n--{boundary}--\r\n".encode())
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        # Groq стоит за Cloudflare, стандартный User-Agent urllib получает 403.
        "User-Agent": "call-transcript/1.0 (+claude-code; python-urllib)",
    }
    for attempt in range(4):
        try:
            with urlopen(Request(ENDPOINT, data=buf.getvalue(), headers=headers, method="POST"),
                         timeout=600, context=ssl.create_default_context()) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code == 429:
                header = exc.headers.get("Retry-After")
                try:
                    wait = float(header or 30)
                except ValueError:
                    wait = 30.0
                if not math.isfinite(wait):
                    wait = 30.0
                if wait <= 120 and attempt < 2:
                    print(f"{path.name}: Groq просит подождать {wait:.0f} с")
                    time.sleep(wait)
                    continue
                # Часовой или дневной лимит: ждать бессмысленно, следующие команды тоже не пустит.
                asks = f", просит ждать {wait:.0f} с" if header else ""
                raise GroqFailed(f"Groq упёрся в лимит (429{asks}): {detail}",
                                 pause=min(max(wait, 600.0), 3600.0))
            if exc.code >= 500 and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            if exc.code >= 500:
                raise GroqFailed(f"Groq ответил {exc.code}: {detail}", pause=300.0)
            if exc.code in (401, 403):
                raise GroqFailed(f"Groq не принял ключ ({exc.code}): {detail}", pause=3600.0)
            # Остальные 4xx - ошибка самого запроса (язык, файл): локальный движок её не исправит.
            raise SystemExit(f"Groq ответил {exc.code} на {path.name}: {detail}")
        except (URLError, TimeoutError, ConnectionError) as exc:
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise GroqFailed(f"сеть не пускает в Groq: {exc}", pause=300.0)
    raise GroqFailed("Groq не ответил", pause=300.0)


def groq_paused(work: Path) -> str | None:
    """Причина паузы Groq, если она ещё идёт. Испорченный файл паузы не мешает: паузы нет."""
    path = work / GROQ_PAUSE_FILE
    if not path.exists():
        return None
    try:
        pause = json.loads(path.read_text(encoding="utf-8"))
        until = float(pause["until"])
        reason = str(pause.get("reason") or "Groq на паузе")
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if not math.isfinite(until) or time.time() >= until:
        return None
    return f"{reason}; пауза ещё {(until - time.time()) / 60:.0f} мин"


def pause_groq(work: Path, seconds: float, reason: str) -> None:
    (work / GROQ_PAUSE_FILE).write_text(
        json.dumps({"until": time.time() + seconds, "reason": reason}, ensure_ascii=False), encoding="utf-8")


def from_whisper_cpp(data: dict, language: str | None = None) -> dict:
    """Ответ whisper.cpp (-ojf) в виде verbose_json Groq: дальше скилл разницы не видит.

    Уверенность - средний логарифм вероятностей токенов без служебных; она выходит выше,
    чем у Groq, поэтому «низкая уверенность» в check на таких кусках срабатывает реже.
    no_speech_prob у whisper.cpp нет, ставим 0. «Повторы» считаются как у Whisper,
    но по сегменту, а не по 30-секундному окну, поэтому «повтор внутри» почти не срабатывает.
    Язык при заданном language - он сам: на тишине whisper.cpp пишет en даже при -l ru.
    """
    segments = []
    for s in data.get("transcription") or []:
        text = str(s.get("text", ""))
        tokens = [t for t in s.get("tokens") or [] if not str(t.get("text", "")).startswith("[_")]
        logprob = (sum(math.log(max(float(t.get("p", 0.0)), 1e-6)) for t in tokens) / len(tokens)
                   if tokens else 0.0)
        raw = text.strip().encode("utf-8")
        segments.append({
            "start": s["offsets"]["from"] / 1000, "end": s["offsets"]["to"] / 1000, "text": text,
            "avg_logprob": logprob, "no_speech_prob": 0.0,
            "compression_ratio": len(raw) / len(zlib.compress(raw)) if raw else 0.0,
        })
    code = language or str((data.get("result") or {}).get("language") or "")
    return {"engine": "local", "language": LANG_NAMES.get(code, code), "segments": segments,
            "text": "".join(s["text"] for s in segments)}


def local_problem() -> str | None:
    """Почему запасной движок сейчас недоступен, или None."""
    if shutil.which(WHISPER_CPP) is None:
        return f"нет {WHISPER_CPP} (brew install whisper-cpp)"
    if not LOCAL_MODEL.exists():
        return f"нет модели {LOCAL_MODEL} (как скачать - в README скилла)"
    return None


def local_whisper(path: Path, language: str | None, work: Path) -> dict:
    """Распознать файл локально: whisper.cpp и large-v3.

    Без поиска речи Silero: он вырезает паузы, и речь до и после паузы становится одним
    сегментом, время которого накрывает паузу. check помечает такой сегмент «на тишине», и
    настоящая речь удаляется. Выдумки на тишине без Silero ловит check, как у Groq.
    """
    problem = local_problem()
    if problem:
        raise SystemExit(f"запасной движок: {problem}")
    # Временная папка внутри рабочей: если команду оборвут, текст созвона не останется в /tmp.
    with tempfile.TemporaryDirectory(dir=work, prefix=".whisper-") as tmp:
        base = Path(tmp) / "out"
        # -ml 120 -sow: сегменты около 6 с, как у Groq; без них по 10-20 с и две реплики в одном.
        cmd = [WHISPER_CPP, "-m", str(LOCAL_MODEL), "-l", language or "auto", "-t", str(min(8, os.cpu_count() or 4)),
               "-ml", "120", "-sow", "-sns", "-ojf", "-of", str(base), "-np"]
        if language in LOCAL_PROMPTS:
            cmd += ["--prompt", LOCAL_PROMPTS[language]]
        try:
            result = subprocess.run([*cmd, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    timeout=1800)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"запасной движок: whisper-cli не уложился в 30 минут на {path.name}") from None
        error = result.stderr.decode("utf-8", "replace").strip()[-400:]
        if result.returncode:
            raise SystemExit(f"запасной движок: whisper-cli {result.returncode} на {path.name}: {error}")
        out = base.with_suffix(".json")
        if not out.exists():
            # Так бывает при неизвестном коде языка: whisper-cli печатает справку и выходит с 0.
            raise SystemExit(f"запасной движок: whisper-cli не выдал результат на {path.name} "
                             f"(язык «{language or 'auto'}»?): {error}")
        # Токен иногда режет букву пополам, и в JSON попадает битый UTF-8.
        data = json.loads(out.read_bytes().decode("utf-8", "replace"))
    return from_whisper_cpp(data, language)


def locally(path: Path, language: str | None, work: Path, reason: str) -> dict:
    print(f"{path.name}: {reason}, распознаю локально")
    data = local_whisper(path, language, work)
    # Причина остаётся в кеше: шапку могут писать в другой сессии.
    data["fallback_reason"] = reason
    return data


def recognize(key: str, path: Path, language: str | None, work: Path) -> dict:
    """Groq, а если он не дал ответа - локально. Режим задаёт CALL_TRANSCRIPT_ENGINE."""
    if ENGINE not in ("auto", "groq", "local"):
        raise SystemExit(f"CALL_TRANSCRIPT_ENGINE={ENGINE!r}: нужно auto, groq или local")
    if ENGINE == "local":
        return locally(path, language, work, "CALL_TRANSCRIPT_ENGINE=local")
    problem = local_problem() if ENGINE == "auto" else "режим groq"
    if not problem:
        reason = groq_paused(work) or ("нет ключа Groq" if not key else None)
        if reason:
            return locally(path, language, work, reason)
    try:
        return whisper(key, path, language)
    except GroqFailed as exc:
        if problem:
            # Запасного пути нет: паузу не ставим, иначе следующие команды не попробуют даже Groq.
            raise SystemExit(f"{path.name}: {exc.reason}"
                             + ("" if ENGINE == "groq" else f"; запасной движок недоступен: {problem}")) from None
        if exc.pause:
            pause_groq(work, exc.pause, exc.reason)
        return locally(path, language, work, exc.reason)


def cached_whisper(key: str, clip: Path, target: Path, language: str | None, work: Path) -> dict:
    if not target.exists():
        data = recognize(key, clip, language, work)
        target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        if data.get("engine") != "local":
            time.sleep(2)
    return json.loads(target.read_text(encoding="utf-8"))


def local_mark(data: dict) -> str:
    return ", локально" if data.get("engine") == "local" else ""


def ffmpeg(*args: str) -> None:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                            capture_output=True, text=True)
    if result.returncode:
        raise SystemExit("ffmpeg: " + result.stderr.strip()[-400:])


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True).stdout
    return float(out.strip())


def cut(audio: Path, target: Path, start: float, length: float) -> None:
    ffmpeg("-ss", f"{start:.3f}", "-i", str(audio), "-t", f"{length:.3f}", "-c", "copy", str(target))


def read_silences(work: Path) -> list[tuple[float, float]]:
    path = work / "silences.tsv"
    if not path.exists():
        raise SystemExit(f"нет {path}: сначала prepare")
    return [tuple(map(float, line.split("\t")[:2])) for line in path.read_text().splitlines() if line]


def silence_share(silences: list[tuple[float, float]], a: float, b: float) -> float:
    if b <= a:
        return 1.0
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in silences) / (b - a)


def read_rows(work: Path, tag: str) -> list[dict]:
    path = work / f"segments_{tag}.tsv"
    if not path.exists():
        raise SystemExit(f"нет {path}: сначала transcribe --tag {tag}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        start, end, nsp, lp, cr, text = line.split("\t")
        rows.append({"start": float(start), "end": float(end), "no_speech_prob": float(nsp),
                     "avg_logprob": float(lp), "compression_ratio": float(cr), "text": text})
    return rows


def cmd_prepare(a: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("нет ffmpeg: brew install ffmpeg (Linux: sudo apt install ffmpeg)")
    media, work = Path(a.media).resolve(), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    audio = work / "audio.mp3"
    streams = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                              "-of", "csv=p=0", str(media)], capture_output=True, text=True).stdout.split()
    if not streams:
        raise SystemExit(f"в {media.name} нет звуковой дорожки")
    if not audio.exists():
        mix = []
        if len(streams) > 1:
            # Без смешивания ffmpeg возьмёт одну дорожку, и голос с другой пропадёт целиком.
            labels = "".join(f"[0:a:{i}]" for i in range(len(streams)))
            mix = ["-filter_complex", f"{labels}amix=inputs={len(streams)}:duration=longest[a]", "-map", "[a]"]
        ffmpeg("-i", str(media), *mix, "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
               str(audio))
    total = duration(audio)

    log = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(audio), "-af",
                          "silencedetect=noise=-40dB:d=0.8", "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = [max(0.0, float(x)) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)] + [total]
    silences = list(zip(starts, ends))
    (work / "silences.tsv").write_text("".join(f"{s:.2f}\t{e:.2f}\t{e - s:.2f}\n" for s, e in silences))

    cuts, pos = [0.0], 0.0
    while total - pos > CHUNK_SECONDS * 1.3:
        target = pos + CHUNK_SECONDS
        near = [(s + e) / 2 for s, e in silences if abs((s + e) / 2 - target) <= 90]
        pos = min(near, key=lambda m: abs(m - target)) if near else target
        cuts.append(pos)
    cuts.append(total)
    chunks = work / "chunks"
    chunks.mkdir(exist_ok=True)
    lines = []
    for i, (s, e) in enumerate(zip(cuts, cuts[1:])):
        cut(audio, chunks / f"chunk_{i:03d}.mp3", s, e - s)
        lines.append(f"{i}\t{s:.3f}\t{e - s:.3f}\n")
    (work / "chunks.tsv").write_text("".join(lines))
    (work / "meta.json").write_text(json.dumps({"media": str(media), "duration": total, "audio_streams": len(streams)},
                                               ensure_ascii=False))

    print(f"длительность {stamp(total)}, кусков {len(lines)}, звуковых дорожек {len(streams)}")
    if len(streams) > 1:
        print("дорожек несколько: они смешаны в одну. Если на каждой свой человек, говорящих можно разделить по дорожкам")
    long = [(s, e) for s, e in silences if e - s >= LONG_PAUSE]
    print("паузы от 6 с:", ", ".join(f"{stamp(s)}–{stamp(e)}" for s, e in long) or "нет")


def cmd_langs(a: argparse.Namespace) -> None:
    work, key = Path(a.work), load_key()
    total = json.loads((work / "meta.json").read_text())["duration"]
    silences = read_silences(work)
    out = work / "langs"
    out.mkdir(exist_ok=True)
    counts = Counter()
    for start in range(int(a.first), int(total - a.length), a.every):
        if silence_share(silences, start, start + a.length) > 0.5:
            continue
        clip = out / f"w_{start:05d}.mp3"
        if not clip.exists():
            cut(work / "audio.mp3", clip, start, a.length)
        data = cached_whisper(key, clip, clip.with_suffix(".json"), None, work)
        lang = str(data.get("language"))
        counts[lang] += 1
        text = " ".join(s["text"].strip() for s in data.get("segments") or [])
        print(f"{stamp(start)} {lang}{local_mark(data)}: {text[:100]}")
    print("итого:", ", ".join(f"{lang} {n}" for lang, n in counts.most_common()))


def cmd_transcribe(a: argparse.Namespace) -> None:
    work, key = Path(a.work), load_key()
    tag = a.tag or a.lang
    language = None if a.lang == "auto" else a.lang
    raw = work / "raw" / tag
    raw.mkdir(parents=True, exist_ok=True)
    rows, local, reasons = [], [], []
    for line in (work / "chunks.tsv").read_text().splitlines():
        idx, offset, length = line.split("\t")
        name = f"chunk_{int(idx):03d}"
        data = cached_whisper(key, work / "chunks" / f"{name}.mp3", raw / f"{name}.json", language, work)
        segs = data.get("segments") or []
        empty = sum(1 for s in segs if not str(s.get("text", "")).strip())
        print(f"{name}: язык {data.get('language')}, сегментов {len(segs)}, пустых {empty}{local_mark(data)}")
        if data.get("engine") == "local":
            local.append(f"{stamp(float(offset))}–{stamp(float(offset) + float(length))}")
            reason = data.get("fallback_reason")
            if reason and reason not in reasons:
                reasons.append(reason)
        for s in segs:
            text = re.sub(r"\s+", " ", str(s.get("text", ""))).strip()
            rows.append((float(offset) + s["start"], float(offset) + s["end"], s.get("no_speech_prob", 0.0),
                         s.get("avg_logprob", 0.0), s.get("compression_ratio", 0.0), text))
    (work / f"segments_{tag}.tsv").write_text(
        "".join(f"{s:.2f}\t{e:.2f}\t{n:.3f}\t{lp:.3f}\t{cr:.3f}\t{t}\n" for s, e, n, lp, cr, t in rows), encoding="utf-8")
    (work / f"segments_{tag}.txt").write_text(
        "".join(f"{i} [{stamp(r[0])}] {r[5] or '—'}\n" for i, r in enumerate(rows, 1)), encoding="utf-8")
    print(f"сегментов {len(rows)}: {work / f'segments_{tag}.txt'}")
    if local:
        print("распознано локально (whisper.cpp), а не в Groq:", ", ".join(local))
        if reasons:
            print("причина:", "; ".join(reasons))


def cmd_check(a: argparse.Namespace) -> None:
    work = Path(a.work)
    rows, silences = read_rows(work, a.tag), read_silences(work)
    total = json.loads((work / "meta.json").read_text())["duration"]
    long = [(s, e) for s, e in silences if e - s >= LONG_PAUSE]
    prev_start, prev_end, flagged = -1.0, 0.0, 0
    for n, r in enumerate(rows, 1):
        text, reasons = r["text"], []
        if not text:
            reasons.append("пусто")
        if r["start"] < prev_start:
            reasons.append("время назад")
        gap = r["start"] - prev_end
        if gap > a.gap and silence_share(silences, prev_end, r["start"]) < 0.5:
            reasons.append(f"перед ним {gap:.0f} с без текста, а тишины там нет")
        length, words = r["end"] - r["start"], len(text.split())
        if text and any(s <= (r["start"] + r["end"]) / 2 <= e for s, e in long):
            reasons.append("на тишине")
        elif any(s + 3 < r["start"] < e < r["end"] for s, e in long):
            reasons.append("начало внутри паузы, время съехало")
        if any(p in text.lower() for p in HALLUCINATIONS):
            reasons.append("типичная выдумка Whisper")
        if text and length >= 8 and words / length < 0.35:
            reasons.append(f"{words} сл. на {length:.0f} с, речь пропущена")
        if text and r["avg_logprob"] < -0.6:
            reasons.append("низкая уверенность")
        if r["compression_ratio"] > 2.4:
            reasons.append("повтор внутри")
        if n >= 3 and text and text == rows[n - 2]["text"] == rows[n - 3]["text"]:
            reasons.append("третий повтор подряд")
        if text and r["no_speech_prob"] > 0.6 and (words <= 2 or r["avg_logprob"] < -1.0):
            reasons.append("похоже на шум")
        if reasons:
            flagged += 1
            print(f"{n} [{stamp(r['start'])}] {'; '.join(reasons)}: {text[:90]}")
        prev_start, prev_end = r["start"], max(prev_end, r["end"])
    if total - prev_end > a.gap and silence_share(silences, prev_end, total) < 0.5:
        print(f"конец [{stamp(prev_end)}–{stamp(total)}] без текста, а тишины там нет")
    print(f"отмечено сегментов: {flagged} из {len(rows)}")


def cmd_compare(a: argparse.Namespace) -> None:
    """Два прогона вперемешку по времени: где один потерял или выровнял речь в свой язык, видно сразу."""
    work = Path(a.work)
    main, other = read_rows(work, a.tag), read_rows(work, a.other)
    lo, hi = (int(x) for x in a.range.split("-"))
    lo, hi = max(1, lo), min(len(main), hi)
    t0, t1 = main[lo - 1]["start"], max(r["end"] for r in main[lo - 1:hi])
    pad = " " * len(str(hi))
    lines = [(r["start"], 0, f"{n} [{stamp(r['start'])}] {a.tag}  {r['text'] or '—'}")
             for n, r in enumerate(main, 1) if lo <= n <= hi]
    lines += [(r["start"], 1, f"{pad} [{stamp(r['start'])}] {a.other}  {r['text'] or '—'}")
              for r in other if t0 - 2 <= r["start"] < t1]
    for _, _, line in sorted(lines):
        print(line)


def cmd_clip(a: argparse.Namespace) -> None:
    work, key = Path(a.work), load_key()
    out = work / "clips"
    out.mkdir(exist_ok=True)
    for spec in a.starts:
        start = parse_time(spec)
        name = f"c_{int(start):05d}_{int(a.len)}"
        clip = out / f"{name}.mp3"
        if not clip.exists():
            cut(work / "audio.mp3", clip, start, a.len)
        for lang in a.langs.split(","):
            data = cached_whisper(key, clip, out / f"{name}.{lang}.json", None if lang == "auto" else lang, work)
            print(f"== {stamp(start)} {lang} (Whisper: {data.get('language')}{local_mark(data)})")
            for s in data.get("segments") or []:
                print(f"{stamp(start + s['start'])} {s['text'].strip()}")


def load_soniox_key() -> str:
    key = setting("SONIOX_API_KEY")
    if not key:
        raise SystemExit(f"нет SONIOX_API_KEY ни в окружении, ни в {ENV_FILE}: быстрый режим недоступен")
    return key


# Сбои связи с Soniox: urllib, ssl и http.client бросают их в разных местах ответа.
NET_ERRORS = (OSError, http.client.HTTPException)


def soniox_call(key: str, method: str, path: str, *, body: bytes | None = None,
                content_type: str | None = None, timeout: float = 120) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {key}", "User-Agent": "call-transcript/1.0 (+claude-code; python-urllib)"}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(SONIOX_API + path, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
            status, raw = response.status, response.read()
    except HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        data = json.loads(raw) if raw else {}
    except ValueError:
        # Например, HTML от прокси вместо JSON.
        data = {"message": raw[:300].decode("utf-8", "replace")}
    return status, data if isinstance(data, dict) else {"items": data}


def soniox_request(key: str, method: str, path: str, *, attempts: int = 4, **kwargs) -> tuple[int, dict]:
    """Запрос с повторами на разовые сбои: сеть, 429, 5xx. Загрузку и задание не повторяем."""
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            status, data = soniox_call(key, method, path, **kwargs)
        except NET_ERRORS:
            if last:
                raise
            time.sleep(3 * (attempt + 1))
            continue
        if (status == 429 or status >= 500) and not last:
            time.sleep(3 * (attempt + 1))
            continue
        return status, data
    raise AssertionError("недостижимо")


def soniox_ok(status: int, data: dict, what: str) -> dict:
    if status < 400:
        return data
    if status == 402:
        raise SystemExit("Soniox: баланс пуст (402), пополни на console.soniox.com")
    raise SystemExit(f"Soniox: {what} - {status} {data.get('error_type', '')}: {data.get('message', '')}")


def soniox_warn(kind: str, ident: str, why: str) -> None:
    print(f"ВНИМАНИЕ: у Soniox осталось {kind} {ident} ({why}). Удали в console.soniox.com")


def soniox_delete(key: str, kind: str, ident: str) -> bool:
    try:
        for attempt in range(40):
            status, _data = soniox_request(key, "DELETE", f"/{kind}/{ident}")
            if status != 409:
                break
            if attempt == 0:
                print("Soniox ещё распознаёт, жду, чтобы удалить (до 2 минут)")
            time.sleep(3)
        check, _data = soniox_request(key, "GET", f"/{kind}/{ident}")
    except Exception as exc:  # noqa: BLE001 - любой сбой здесь - громкое предупреждение, а не тихий выход
        soniox_warn(kind, ident, type(exc).__name__)
        return False
    if check == 404:
        print(f"у Soniox удалено: {kind}")
        return True
    soniox_warn(kind, ident, f"удаление {status}, проверка {check}")
    return False


def soniox_find(key: str, ref: str) -> list[tuple[str, str]]:
    """Файлы и транскрипции с нашей меткой: так находятся те, чей ответ потерялся."""
    found = []
    for kind in ("transcriptions", "files"):
        try:
            status, data = soniox_request(key, "GET", f"/{kind}")
        except Exception as exc:  # noqa: BLE001
            print(f"ВНИМАНИЕ: не удалось проверить список {kind} у Soniox ({type(exc).__name__})")
            continue
        items = data.get(kind) if status < 400 else None
        if not isinstance(items, list):
            print(f"ВНИМАНИЕ: не удалось проверить список {kind} у Soniox (код {status})")
            continue
        found += [(kind, str(i["id"])) for i in items if isinstance(i, dict) and i.get("client_reference_id") == ref]
    return found


def soniox_forget(key: str, ids: dict) -> bool:
    """Удалить у Soniox транскрипцию и файл этого запуска и проверить, что их нет.

    Удаляет и известные по ответам, и найденные по метке (ответ мог потеряться). Пока идёт
    удаление, Ctrl+C и SIGTERM не прерывают его. Транскрипцию, которая ещё распознаётся,
    Soniox не удаляет (409): ждём. True - у Soniox ничего не осталось.
    """
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in saved:
        signal.signal(sig, signal.SIG_IGN)
    try:
        targets = [("transcriptions", ids.get("transcription_id")), ("files", ids.get("file_id"))]
        if ids.get("ref"):
            targets += [t for t in soniox_find(key, ids["ref"]) if t not in targets]
        ok = True
        for kind, ident in targets:
            if ident:
                ok = soniox_delete(key, kind, str(ident)) and ok
        return ok
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def soniox_transcribe(key: str, audio: Path, langs: list[str], terms: list[str], pending: Path) -> dict:
    """Загрузить звук, распознать с говорящими, забрать текст. У Soniox ничего не остаётся.

    До загрузки в pending пишется метка запуска, потом номера файла и задания: если процесс
    убьют наглухо, следующий запуск по ним уберёт всё у Soniox.
    """
    ids = {"ref": f"call-transcript-{uuid.uuid4().hex}", "file_id": None, "transcription_id": None}

    def remember() -> None:
        pending.write_text(json.dumps(ids), encoding="utf-8")

    remember()
    started = time.time()
    try:
        boundary = f"----call{uuid.uuid4().hex}"
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="client_reference_id"\r\n\r\n'
                f'{ids["ref"]}\r\n'
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{audio.name}"\r\n'
                f"Content-Type: audio/mpeg\r\n\r\n").encode() + audio.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        ids["file_id"] = soniox_ok(*soniox_call(key, "POST", "/files", body=body, timeout=600,
                                                content_type=f"multipart/form-data; boundary={boundary}"),
                                   "загрузка")["id"]
        remember()
        uploaded = time.time()
        payload = {
            "model": SONIOX_MODEL, "file_id": ids["file_id"], "language_hints": langs,
            "enable_speaker_diarization": True, "enable_language_identification": True,
            "context": {"general": [{"key": "speakers", "value": "video call"}], "terms": terms},
            "client_reference_id": ids["ref"],
        }
        ids["transcription_id"] = soniox_ok(*soniox_call(key, "POST", "/transcriptions",
                                                         body=json.dumps(payload).encode(),
                                                         content_type="application/json"), "задание")["id"]
        remember()
        while True:
            state = soniox_ok(*soniox_request(key, "GET", f"/transcriptions/{ids['transcription_id']}"), "статус")
            if state.get("status") == "completed":
                break
            if state.get("status") == "error":
                raise SystemExit(f"Soniox: распознавание не удалось: {state.get('error_message', '')}")
            if time.time() - uploaded > SONIOX_WAIT:
                raise SystemExit(f"Soniox: не уложился в {SONIOX_WAIT // 60} минут")
            time.sleep(3)
        transcript = soniox_ok(*soniox_request(key, "GET", f"/transcriptions/{ids['transcription_id']}/transcript",
                                               timeout=300), "текст")
        print(f"Soniox: загрузка {uploaded - started:.0f} с, распознавание {time.time() - uploaded:.0f} с")
        return transcript
    except NET_ERRORS as exc:
        raise SystemExit(f"Soniox: сеть: {type(exc).__name__}: {exc}") from None
    finally:
        if soniox_forget(key, ids):
            pending.unlink(missing_ok=True)


def soniox_segments(tokens: list[dict], pause: float = 1.5, soft: float = 10.0, hard: float = 20.0) -> list[dict]:
    """Токены Soniox в сегменты: новый сегмент при смене говорящего, паузе, конце длинной фразы.

    Режем только на границе слова (токен с пробелом в начале): смена говорящего посреди слова
    или на точке остаётся у прежнего говорящего, слово не рвётся. Токен без говорящего - у
    прежнего. Токены из одних пробелов не двигают время, чтобы не прятать паузу.
    """
    segments: list[dict] = []
    current: dict | None = None
    for t in tokens:
        text = str(t.get("text") or "")
        if not text:
            continue
        raw_speaker = t.get("speaker")
        speaker = (str(raw_speaker) if raw_speaker not in (None, "")
                   else current["speaker"] if current is not None else "1")
        start, end = t["start_ms"] / 1000, t["end_ms"] / 1000
        blank = not text.strip()
        if current is not None and not blank:
            new_word = text.startswith(" ")
            length = end - current["start"]
            ends_phrase = "".join(current["parts"]).rstrip().rstrip("»\"'”)").endswith((".", "?", "!", "…"))
            if new_word and (speaker != current["speaker"]
                             or start - current["end"] > pause
                             or (length > soft and ends_phrase)
                             or length > hard):
                segments.append(current)
                current = None
        if current is None:
            if blank:
                continue
            current = {"start": start, "end": end, "speaker": speaker, "parts": [], "conf": [], "langs": Counter()}
        current["parts"].append(text)
        if blank:
            continue
        current["end"] = end
        current["conf"].append(float(t.get("confidence", 1.0)))
        if t.get("language"):
            current["langs"][t["language"]] += 1
    if current is not None:
        segments.append(current)
    out = []
    for s in segments:
        conf = sum(s["conf"]) / len(s["conf"])
        out.append({"start": s["start"], "end": s["end"], "speaker": s["speaker"],
                    "text": re.sub(r"\s+", " ", "".join(s["parts"])).strip(),
                    "confidence": conf, "weakest": min(s["conf"]),
                    "language": s["langs"].most_common(1)[0][0] if s["langs"] else ""})
    return out


def cmd_soniox(a: argparse.Namespace) -> None:
    work = Path(a.work)
    audio = work / "audio.mp3"
    if not audio.exists():
        raise SystemExit(f"нет {audio}: сначала prepare")
    raw = work / "raw" / "soniox"
    raw.mkdir(parents=True, exist_ok=True)
    cache, pending = raw / "transcript.json", raw / "pending.json"
    key = None
    if pending.exists():
        # Прошлый запуск убили, не дав удалить: сначала убираем его у Soniox.
        key = load_soniox_key()
        print("прошлый запуск оборвался: убираю его загрузку у Soniox")
        try:
            ids = json.loads(pending.read_text(encoding="utf-8"))
        except ValueError:
            ids = {}
        if soniox_forget(key, ids if isinstance(ids, dict) else {}):
            pending.unlink(missing_ok=True)
    if not cache.exists():
        extra = f"{setting('CALL_TRANSCRIPT_TERMS')},{a.terms or ''}"
        terms = list(dict.fromkeys(SONIOX_TERMS + [t.strip() for t in extra.split(",") if t.strip()]))
        transcript = soniox_transcribe(key or load_soniox_key(), audio, a.langs.split(","), terms, pending)
        # Целиком или никак: оборванная запись не должна стать «кешем».
        part = cache.with_suffix(".part")
        part.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        part.replace(cache)
    tokens = json.loads(cache.read_text(encoding="utf-8")).get("tokens") or []
    segs = soniox_segments(tokens)
    if not segs:
        raise SystemExit("Soniox не вернул текста: в записи нет речи или звук испорчен")
    # Тот же вид, что у Groq: check, compare и assemble работают без изменений.
    (work / "segments_soniox.tsv").write_text("".join(
        f"{s['start']:.2f}\t{s['end']:.2f}\t0.000\t{math.log(max(s['confidence'], 1e-6)):.3f}\t0.000\t{s['text']}\n"
        for s in segs), encoding="utf-8")
    (work / "segments_soniox.txt").write_text("".join(
        f"{i} [{stamp(s['start'])}] ({s['speaker']}) {s['text'] or '—'}\n" for i, s in enumerate(segs, 1)),
        encoding="utf-8")
    (work / "speakers_soniox.json").write_text(json.dumps([s["speaker"] for s in segs]), encoding="utf-8")
    words = Counter()
    for s in segs:
        words[s["speaker"]] += len(s["text"].split())
    langs = Counter(s["language"] for s in segs)
    total = max(1, sum(words.values()))
    print(f"сегментов {len(segs)}: {work / 'segments_soniox.txt'}")
    print("говорящие по голосу:", ", ".join(f"{who} - {n * 100 // total}% слов" for who, n in words.most_common()))
    print("языки сегментов:", ", ".join(f"{lang or '?'} {n}" for lang, n in langs.most_common()))
    # Средняя по сегменту прячет одно сомнительное слово, поэтому смотрим самое слабое.
    shaky = [f"{i} [{stamp(s['start'])}]" for i, s in enumerate(segs, 1) if s["weakest"] < 0.5]
    print(f"сегменты со словом, в котором Soniox не уверен ({len(shaky)}):", ", ".join(shaky[:60]) or "нет")


def cmd_names(a: argparse.Namespace) -> None:
    """Имена говорящим Soniox: turns в edits_<тег>.json со смены говорящего.

    Отдельный файл, а не edits.json: в рабочей папке может лежать edits.json обычного режима
    с номерами сегментов Groq, и его правки легли бы не на те сегменты.
    """
    work = Path(a.work)
    path = work / f"speakers_{a.tag}.json"
    if not path.exists():
        raise SystemExit(f"нет {path}: сначала soniox")
    speakers = json.loads(path.read_text(encoding="utf-8"))
    if not speakers:
        raise SystemExit("сегментов нет: Soniox не вернул текста")
    names = {}
    for pair in a.pairs:
        sid, sep, name = pair.partition("=")
        if not sep or not name.strip():
            raise SystemExit(f"«{pair}»: нужно номер=имя, например 1=Я")
        names[sid.strip()] = name.strip()
    missing = sorted(set(speakers) - set(names))
    if missing:
        raise SystemExit(f"нет имени для говорящих {', '.join(missing)}: перечисли всех, например 3=Собеседник")
    edits_path = Path(a.edits) if a.edits else work / f"edits_{a.tag}.json"
    edits = json.loads(edits_path.read_text(encoding="utf-8")) if edits_path.exists() else {}
    turns, previous = {}, None
    for n, sid in enumerate(speakers, 1):
        if names[sid] != previous:
            turns[str(n)] = names[sid]
            previous = names[sid]
    # Все названные, даже если Soniox свёл два голоса в один: иначе turns на них не пройдут assemble.
    edits["speakers"] = list(dict.fromkeys([names[s] for s in speakers] + list(names.values())))
    edits["turns"] = turns
    edits_path.write_text(json.dumps(edits, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"смен говорящего {len(turns)}, говорящие {', '.join(edits['speakers'])}: {edits_path}")


def cmd_assemble(a: argparse.Namespace) -> None:
    work = Path(a.work)
    rows = read_rows(work, a.tag)
    edits = json.loads(Path(a.edits).read_text(encoding="utf-8"))
    speakers = edits.get("speakers") or []

    def seg(n) -> dict:
        n = int(n)
        if not 1 <= n <= len(rows):
            raise SystemExit(f"сегмента {n} нет, всего {len(rows)}")
        return rows[n - 1]

    for n, old, new in edits.get("fix", []):
        r = seg(n)
        if old not in r["text"]:
            raise SystemExit(f"fix {n}: нет «{old}» в «{r['text']}»")
        r["text"] = r["text"].replace(old, new)
    for n, start in edits.get("start", {}).items():
        seg(n)["start"] = float(start)
    for n, text in edits.get("set", {}).items():
        seg(n)["text"] = text or ""
    for n, note in edits.get("notes", {}).items():
        seg(n)["note"] = note
    hits = Counter()
    for r in rows:
        for old, new in edits.get("global", []):
            hits[old] += r["text"].count(old)
            r["text"] = r["text"].replace(old, new)

    turns = {int(n): who for n, who in edits.get("turns", {}).items()}
    splits = {int(n): (marker, who) for n, marker, who in edits.get("split", [])}
    if 1 not in turns:
        raise SystemExit("turns должен называть говорящего с сегмента 1")
    for who in list(turns.values()) + [who for _, who in splits.values()]:
        if speakers and who not in speakers:
            raise SystemExit(f"«{who}» нет в speakers {speakers}")

    # Говорящий тянется по всем сегментам, включая удалённые, чтобы смена на удалённом не потерялась.
    speaker, items = None, []
    for n, r in enumerate(rows, 1):
        seg(n)
        speaker = turns.get(n, speaker)
        if "note" in r:
            items.append({"n": n, "note": r["note"], "start": r["start"]})
            continue
        text = r["text"].strip()
        if n in splits:
            marker, nxt = splits[n]
            head, sep, tail = text.partition(marker)
            if not sep:
                raise SystemExit(f"split {n}: нет «{marker}» в «{text}»")
            if head.strip():
                items.append({"n": n, "start": r["start"], "end": r["end"], "text": head.strip(), "speaker": speaker})
            speaker, text = nxt, (sep + tail).strip()
        if text:
            items.append({"n": n, "start": r["start"], "end": r["end"], "text": text, "speaker": speaker})

    if a.preview:
        lo, hi = (int(x) for x in a.preview.split("-"))
        for it in items:
            if lo <= it["n"] <= hi:
                who = "заметка" if "note" in it else it["speaker"]
                print(f"{it['n']} [{stamp(it['start'])}] {who}: {it.get('note') or it['text']}")
        return

    # Реплика = подряд идущие сегменты одного говорящего; длинную режем на абзацы.
    body, current, first, words = [], [], True, Counter()

    def flush(first_of_turn: bool) -> None:
        text = re.sub(r"\s+", " ", " ".join(it["text"] for it in current)).strip()
        label = f" {current[0]['speaker']}:" if first_of_turn else ""
        body.append(f"**[{stamp(current[0]['start'])}]{label}** {text}")
        words[current[0]["speaker"]] += len(text.split())

    for it in items:
        if "note" in it:
            if current:
                flush(first)
            body.append(f"_… {it['note']} …_")
            current, first = [], True
            continue
        if current and it["speaker"] != current[-1]["speaker"]:
            flush(first)
            current, first = [], True
        elif current and it["end"] - current[0]["start"] > edits.get("paragraph_seconds", 80):
            flush(first)
            current, first = [], False
        current.append(it)
    if current:
        flush(first)

    out = Path(a.out).expanduser()
    header = Path(a.header).read_text(encoding="utf-8").rstrip("\n")
    out.write_text(header + "\n\n" + "\n\n".join(body) + "\n", encoding="utf-8")
    replies = sum(1 for b in body if re.match(r"\*\*\[[\d:]+\] [^*]+:\*\*", b))
    share = ", ".join(f"{who} {n * 100 // max(1, sum(words.values()))}%" for who, n in words.most_common())
    print(f"реплик {replies}, абзацев {len(body)}, доля слов: {share}")
    unused = [old for old, _ in edits.get("global", []) if not hits[old]]
    if unused:
        print("global без совпадений:", ", ".join(unused))
    print(out)


def stop_on_signal(signum: int, _frame) -> None:
    """SIGTERM как Ctrl+C: выход через SystemExit, чтобы finally успели убрать за собой у Soniox."""
    raise SystemExit(128 + signum)


def main() -> None:
    signal.signal(signal.SIGTERM, stop_on_signal)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="звук, паузы и куски")
    s.add_argument("media")
    s.add_argument("work")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("langs", help="язык по окнам")
    s.add_argument("work")
    s.add_argument("--every", type=int, default=180, help="шаг окон, с")
    s.add_argument("--length", type=float, default=30, help="длина окна, с")
    s.add_argument("--first", type=float, default=30, help="начало первого окна, с")
    s.set_defaults(func=cmd_langs)

    s = sub.add_parser("transcribe", help="прогон кусков")
    s.add_argument("work")
    s.add_argument("--lang", required=True, help="код языка (uk, ru, en) или auto")
    s.add_argument("--tag", help="имя прогона, по умолчанию код языка")
    s.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("check", help="места под ручную проверку")
    s.add_argument("work")
    s.add_argument("--tag", required=True)
    s.add_argument("--gap", type=float, default=8, help="дыра без текста, с")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("compare", help="два прогона рядом по времени")
    s.add_argument("work")
    s.add_argument("--tag", required=True, help="основной прогон, его номера сегментов")
    s.add_argument("--other", required=True, help="справочный прогон")
    s.add_argument("--range", required=True, help="сегменты основного прогона A-B")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("clip", help="отрывки заново")
    s.add_argument("work")
    s.add_argument("starts", nargs="+", help="начало: ММ:СС, Ч:ММ:СС или секунды")
    s.add_argument("--len", type=float, default=30, help="длина отрывка, с")
    s.add_argument("--langs", default="auto", help="через запятую: uk,ru или auto")
    s.set_defaults(func=cmd_clip)

    s = sub.add_parser("soniox", help="быстрый режим: звук в Soniox, текст и говорящие по голосу")
    s.add_argument("work")
    s.add_argument("--langs", default="ru,uk", help="подсказка языков через запятую")
    s.add_argument("--terms", help="ещё слова для подсказки через запятую: имена, названия")
    s.set_defaults(func=cmd_soniox)

    s = sub.add_parser("names", help="быстрый режим: имена говорящим Soniox в edits.json")
    s.add_argument("work")
    s.add_argument("pairs", nargs="+", help="номер=имя: 1=Я 2=Олег")
    s.add_argument("--tag", default="soniox")
    s.add_argument("--edits", help="по умолчанию <W>/edits_<тег>.json")
    s.set_defaults(func=cmd_names)

    s = sub.add_parser("assemble", help="сборка .md")
    s.add_argument("work")
    s.add_argument("--tag", required=True)
    s.add_argument("--edits", required=True)
    s.add_argument("--header")
    s.add_argument("--out")
    s.add_argument("--preview", help="диапазон сегментов A-B: показать говорящих, файл не писать")
    s.set_defaults(func=cmd_assemble)

    a = p.parse_args()
    if a.cmd == "assemble" and not a.preview and not (a.header and a.out):
        p.error("assemble без --preview требует --header и --out")
    a.func(a)


if __name__ == "__main__":
    main()
