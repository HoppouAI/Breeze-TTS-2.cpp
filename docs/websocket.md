# WebSocket API

The HTTP endpoint already streams audio and already stops generating when the
client disconnects, so the socket exists for the things chunked HTTP cannot do:
feeding text in as it arrives, retuning delivery partway through, interrupting
without tearing the connection down, and holding a session open across many
lines.

cpp-httplib has no WebSocket support and does not hand over the socket after an
upgrade, so this is RFC 6455 implemented directly on its own listener. That is
why it is a second port rather than a route.

## Connecting

The port defaults to the HTTP port plus one. `--ws-port` moves it, `--ws-port -1`
turns it off. `GET /health` reports where it ended up so clients can find it
without being told:

```json
{"status":"ok","sample_rate":24000,"ws_port":8081}
```

```
ws://127.0.0.1:8081
```

**Text frames are JSON control messages, binary frames are audio.** WebSocket
already distinguishes the two at the framing layer, so audio needs no envelope
and no base64. The audio is headerless signed 16 bit little endian mono PCM at
the rate given in the `ready` message, exactly like the HTTP body.

## Messages you send

| `type` | Fields | Meaning |
| --- | --- | --- |
| `start` | `voice_id`, `instruction`, `ref_text`, `cfg_scale`, `seed`, `temperature`, `top_k`, `split_chars` | Opens a session. Encodes the reference once for all of it. |
| `text` | `text` | Adds text. Whole sentences are spoken as they complete, the rest waits. |
| `flush` | `text` | Speaks what is buffered even without a sentence ending. |
| `instruction` | `instruction` | Changes delivery from the next piece onwards. |
| `cancel` | | Stops mid sentence and throws away anything buffered. |
| `end` | `text` | Adds any last text, speaks everything, then reports `done`. |

`start` takes a `voice_id`, not a file. Reference clips are uploaded over HTTP to
`POST /v1/voices`, which hands back an id. See [voices.md](voices.md). Leaving
`voice_id` out gives voice design driven by `instruction`.

## Messages you receive

| `type` | Fields | Meaning |
| --- | --- | --- |
| `ready` | `sample_rate`, `format` | Connection is up. Send `start`. |
| `started` | `voice_id` | Session is open. |
| `speaking` | `text` | The piece about to be generated. |
| `queued` | | Another connection holds the GPU. Waiting, not refused. |
| `instruction_set` | | The new delivery is in effect from the next piece. |
| `cancelled` | | Generation was interrupted. |
| `done` | | Everything buffered has been spoken. |
| `error` | `message` | Something was wrong with the request. |

## A session

```json
-> {"type":"start","voice_id":"harbour","seed":7}
<- {"type":"started","voice_id":"harbour"}
-> {"type":"text","text":"The harbour was quiet that morning. "}
<- {"type":"speaking","text":"The harbour was quiet that morning."}
<- <binary audio>
-> {"type":"instruction","instruction":"Speak in an urgent, alarmed whisper."}
<- {"type":"instruction_set"}
-> {"type":"end","text":"Then the alarm went off."}
<- {"type":"speaking","text":"Then the alarm went off."}
<- <binary audio>
<- {"type":"done"}
```

## Feeding text as it arrives

Text is buffered and drained on sentence boundaries, so a fragment that does not
finish a sentence is held rather than spoken. Sending an LLM's output verbatim as
it streams produces one piece per sentence with no extra work.

If the buffer grows past `split_chars` without a sentence ending, it breaks on a
space instead, so text with no punctuation still gets spoken. `flush` forces
whatever is waiting. It defaults to the server's `--split-chars`, but unlike the
HTTP endpoint it cannot be turned off, since draining needs a size to aim at.
`0` falls back to 600 here.

While there is no reference clip, the opening piece doubles as the reference for
everything after it, so it is capped near a normal clip length. A long first
piece makes the model start skipping sentences later.

## Changing delivery partway

An `instruction` message applies from the **next** piece. Whatever is already
being generated finishes as it was, because a piece is generated as one unit.

The effect is real but it is a nudge, not a transformation. Same sentence, same
seed, measured on the isolated piece:

| Instruction | RMS | Spectral centroid |
| --- | --- | --- |
| `Speak clearly and naturally.` | 0.1438 | 1733 Hz |
| `Speak in a breathy, hushed whisper.` | 0.1126 | 1895 Hz |

Quieter and brighter, which is what a whisper should do. With a cloned reference
the model deliberately protects the reference's timbre, so instructions move
delivery rather than rebuild the voice. Voice design sessions, which have no
reference, respond more strongly.

## Interrupting

`cancel` stops the current piece part way through rather than after it finishes,
and clears anything still buffered. The audio already sent stays valid, so a
client can keep and play what it received.

The session stays open. Send more text and it carries on with the same voice.

## Concurrency

Generation runs one at a time on the GPU. Unlike the HTTP endpoints, which answer
`409` when busy, a socket session **waits its turn** and reports `queued` while it
does. Each piece takes the lock separately, so two connections interleave between
pieces instead of one blocking the other for an entire passage.

Each connection reads and generates on separate threads, which is what lets
`cancel` and `instruction` arrive while audio is still being produced.

## Security

There is no authentication, no origin check and no rate limit on the socket. It
binds to `--host`, which defaults to `127.0.0.1`. Binding it to `0.0.0.0` hands
anyone on the network unrestricted use of the GPU. Put it behind a reverse proxy
that terminates TLS and handles auth before exposing it.

## Client sketch

```python
import asyncio, json, websockets

async def speak(lines):
    async with websockets.connect("ws://127.0.0.1:8081", max_size=None) as ws:
        await ws.recv()  # ready
        await ws.send(json.dumps({"type": "start", "voice_id": "harbour", "seed": 7}))

        async def send():
            for line in lines:
                await ws.send(json.dumps({"type": "text", "text": line}))
            await ws.send(json.dumps({"type": "end", "text": ""}))

        asyncio.create_task(send())
        audio = bytearray()
        async for msg in ws:
            if isinstance(msg, bytes):
                audio.extend(msg)
            elif json.loads(msg)["type"] in ("done", "error"):
                break
        return bytes(audio)
```
