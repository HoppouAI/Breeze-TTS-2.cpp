#pragma once

#include "voices.h"
#include "ws.h"

#include "breeze/codec.h"
#include "breeze/model.h"

#include <mutex>
#include <string>

namespace breeze {

// handles one websocket connection until it closes. generation still serialises on gpu, a
// connection that arrives mid generation waits its turn rather than being turned away
void ws_connection(WsConn & conn, BreezeModel & model, MimiCodec & codec, VoiceStore & store,
                   std::mutex & gpu, int chunk_first, int chunk_max, int split_chars);

} // namespace breeze
