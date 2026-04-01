import asyncio
import websockets
import json
import numpy as np


async def test_inference():
  uri = "ws://localhost:9000"
  async with websockets.connect(uri,max_size=10_000_000) as ws:
    print("Connected Evo1")
    dummy_img = np.zeros((448,448,3),dtype=np.uint8).tolist()
    dummy_state = np.zeros(24).tolist()
    prompt = "Pick up the can and move to the box"
    obs = {
      "image": [dummy_img,dummy_img,dummy_img],
      "image_mask": [1,1,0],
      "state": dummy_state,
      "action_mask": [[1]*24],
      "prompt": prompt
    }
    await ws.send(json.dumps(obs))
    print("Sent data")

    result = await ws.recv()
    action_chunk = json.loads(result)

    print("Action size: {}".format(np.array(action_chunk).shape))

if __name__ == "__main__":
  asyncio.run(test_inference())