import asyncio
import websockets
import numpy as np
import cv2
import threading
import queue
import tempfile
import os
import base64
import io
import time
import ssl
from io import BytesIO


class WebSocketVideoCapture:
    def __init__(self, host='0.0.0.0', port=8001, certfile='cert.pem', keyfile='key.pem'):
        self.host = host
        self.port = port
        #self.certfile = /etc/ssl/private/cert.pem
        #self.keyfile = /etc/ssl/private/key.pem
        self.frame_queue = queue.Queue()
        self.stopped = False
        self.ws_server = None

        # Start the WebSocket server
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.start_server, args=(self.loop,)).start()

    def start_server(self, loop):
        asyncio.set_event_loop(loop)

        # Create SSL context
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile='/root/projectdi/cert.pem', keyfile='/root/projectdi/key.pem')

        # Start the server with SSL context
        self.ws_server = websockets.serve(self.receive_frames, self.host, self.port, ssl=ssl_context)
        loop.run_until_complete(self.ws_server)
        loop.run_forever()

    async def receive_frames(self, websocket, path):
        print("Connection established")
        try:
            while not self.stopped:
                data = await websocket.recv()
                data_stream = BytesIO(data)

                # Read the video frame from the in-memory data stream
                data_stream.seek(0)
                np_arr = np.frombuffer(data_stream.read(), np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    self.frame_queue.put(frame)
                else:
                    print("Received a None frame")

                if cv2.waitKey(25) == ord('q'):
                    break

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed: {e}")

    def read(self):
        if not self.frame_queue.empty():
            return True, self.frame_queue.get()
        return False, None

    def release(self):
        self.stopped = True
        self.loop.stop()
        self.ws_server.ws_server.close()
        self.frame_queue.queue.clear()

    def queue_empty(self):
        return self.frame_queue.empty()

if __name__ == "__main__":
    try:
        # Create an instance of the WebSocketVideoCapture to start the server
        video_capture = WebSocketVideoCapture()
        print("WebSocket video capture server started. Press Ctrl+C to stop.")
        # Keep the main thread alive
        while True:
            pass
    except KeyboardInterrupt:
        print("WebSocket video capture server stopped.")
        video_capture.release()
