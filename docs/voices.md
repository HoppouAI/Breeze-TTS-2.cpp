# Voices

Cloning needs the reference clip run through the codec before anything can be
generated from it. That encode is the single largest part of the wait before the
first audio arrives, and it produces the same result every time, so it is worth
doing once and keeping.

Measured on an RTX 3060 at Q8_0 with a 5.4 s reference:

| Reference supplied as | Reference encode | Time to first audio |
| --- | --- | --- |
| Uploaded WAV | 548 ms | 906 ms |
| Cached voice | 0 ms | 276 ms |

Nothing else about the request changes. The same seed produces byte identical
audio either way, so this only removes repeated work.

## Two kinds of voice

**Cached** voices live in memory. They appear when a clip is posted without a
name, are keyed by a hash of the clip and its transcript, and disappear when the
server restarts. Posting the same clip twice returns the same id and skips the
encode, so a client that registers on every startup costs nothing after the
first time. Raw clips uploaded to `/v1/audio/speech` enter this cache as well:
the handler hashes the upload with its transcript and reuses the stored codes on
a hit, so plain repeated uploads pay the encode only once per clip.

**Saved** voices are files. They are created with a name, written to the voices
folder as `<name>.breeze`, loaded at startup, and never evicted. The CLI, the
server and the web UI all read the same folder, so a voice made in one shows up
in the others.

## Making a saved voice

From the CLI:

```
breeze-cli model.gguf \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --save-voice harbour
```

```
wrote voices/harbour.breeze (67 frames, 5.36 s)
```

Over HTTP, by adding a `name` to a registration:

```
curl -X POST http://127.0.0.1:8137/v1/voices \
  -F "ref_audio=@reference.wav" \
  --form-string "ref_text=This is the exact transcript of the reference audio." \
  --form-string "name=harbour"
```

In the web UI, fill in the reference fields, type a name next to SAVE AS and
press SAVE VOICE. It appears in the dropdown on every panel straight away.

Any of the three can attach a description at the same time, see
[Descriptions and metadata](#descriptions-and-metadata).

Names allow letters, digits, dash and underscore, up to 64 characters. They
become filenames, so anything looser would let a request write outside the
folder.

## Using one

Everywhere a reference clip is accepted, a voice id is accepted instead.

```
breeze-cli model.gguf --voice harbour --text "It is good to hear your voice again."
```

```
curl -X POST http://127.0.0.1:8137/v1/audio/speech \
  --form-string "text=It is good to hear your voice again." \
  --form-string "voice_id=harbour" \
  -o clone.pcm
```

The transcript travels with the voice, so `ref_text` is not needed. Sending one
anyway overrides the stored copy for that request.

An unknown id is a `404`, not a silent fallback to voice design. A typo in a
voice id would otherwise produce a completely different voice and look like a
model problem.

## Listing and removing

```
breeze-cli model.gguf --list-voices
```

```
harbour                    5.36 s  The harbour lights came on one by one as the evening tide began to turn.
                                   Calm older narrator, slow and warm.
young_woman                4.00 s  The harbour lights came on one by one as the evening tide began to turn.
```

A voice with a description gets it on a second line.

This one does not load the model, so it returns immediately.

`GET /v1/voices` returns the same information as JSON, including the cached ones:

```json
[{"id":"harbour","frames":67,"seconds":5.36,"encode_ms":0,"saved":true,"ref_text":"...",
  "description":"Calm older narrator, slow and warm.","metadata":{"accent":"british"}}]
```

`GET /v1/voices/<id>` returns a single voice in the same shape, or `404`.

`DELETE /v1/voices/<id>` drops a voice from memory and answers with
`{"deleted":"harbour","file_kept":true}`. It deliberately does **not** delete the
file, so a saved voice returns on the next restart. Removing a voice for good
means deleting the `.breeze` file yourself.

## Descriptions and metadata

Every voice can carry an optional description, plain text up to 4096 bytes, plus
up to 32 extra fields of your own as name and value pairs. None of it affects
generation. It is there so you can tell your voices apart and so other tools can
filter them.

Set a description when saving:

```
breeze-cli model.gguf --ref-audio reference.wav --ref-text "..." \
  --save-voice harbour --description "Calm older narrator, slow and warm."
```

```
curl -X POST http://127.0.0.1:8137/v1/voices \
  -F "ref_audio=@reference.wav" \
  --form-string "ref_text=This is the exact transcript of the reference audio." \
  --form-string "name=harbour" \
  --form-string "description=Calm older narrator, slow and warm." \
  --form-string 'metadata={"accent":"british","source":"audiobook sample"}'
```

Saving over an existing name keeps its description and fields unless the new
request sends its own.

Change them later with `PATCH /v1/voices/<id>`, which takes JSON:

```
curl -X PATCH http://127.0.0.1:8137/v1/voices/harbour \
  -H "Content-Type: application/json" \
  -d '{"description":"Calm older narrator.","metadata":{"accent":"scottish","source":null}}'
```

Only what you send changes. `metadata` merges into the existing fields, and a
field set to `null` or an empty string is removed. `"description": null` or
`""` clears the description. The answer is the updated voice in the same shape
as `GET`. Form fields work too, with `metadata` sent as a JSON string.

Field names allow letters, digits, dot, dash and underscore, up to 64
characters, and `description` is reserved. Values are strings of up to 1024
bytes. A request that breaks any limit is a `400` and changes nothing.

For a saved voice the `.breeze` file is rewritten straight away. A cached voice
keeps its description in memory only, so it is lost on restart like the voice
itself.

In the web UI the **VOICE LIBRARY** tab lists every voice with its transcript.
Edit the description, add or remove fields with **+ FIELD** and **X**, then press
**SAVE CHANGES** (or Ctrl+Enter in the description). **USE** picks the voice in the
clone tab. The description also shows under the SAVED VOICE dropdown whenever
that voice is selected, and the save row on each panel has an optional
description box.

## The `.breeze` file

A small binary container holding the encoded codes and the transcript. Roughly
800 bytes per second of reference, so a normal clip is a few kilobytes.

| Offset | Size | Contents |
| --- | --- | --- |
| 0 | 4 | Magic `BRZV` |
| 4 | 4 | Format version, 1 or 2 |
| 8 | 4 | Sample rate the codes were made at |
| 12 | 4 | Codebook count |
| 16 | 4 | Frame count |
| 20 | 4 | Transcript length in bytes |
| 24 | n | Transcript, UTF-8 |
| 24+n | | Version 2 only: the description and fields, see below |
| | frames × codebooks × 4 | Codes, little endian int32, frame major |

Version 2 inserts, after the transcript, a 4 byte description length and the
description, a 4 byte field count, and then for each field a 4 byte name length,
the name, a 4 byte value length and the value. All strings are UTF-8 and all
lengths are little endian.

A voice with no description and no fields is still written as version 1, so
files that do not use metadata stay readable by v0.1.2 and older. A version 2
file needs this build or newer. Edits write to a temporary file first and then
replace the original, so a failed save cannot damage an existing voice.

The codebook count is checked against the loaded model on startup. A voice made
for a different model is skipped with a warning rather than loaded into garbage.

These are codes, not audio. They cannot be played, and they are tied to the
codec the model was built with, so treat them as a cache you can rebuild from
the original WAV rather than as an archive format. Keep the WAV.

## Where they live

The default folder is `voices` next to the working directory. Both binaries take
`--voices-dir <path>` to point somewhere else. The folder is created on demand
when the first voice is saved.
