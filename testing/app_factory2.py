import os
import logging
import signal
from flask import Flask, request, jsonify

from background_thread2 import BackgroundThreadFactory, TASKS_QUEUE

logging.basicConfig(level=logging.INFO, force=True)
background_thread = None
skuid_backgroundthread = {}


def create_app():
    app = Flask(__name__)

    @app.route('/task', methods=['POST'])
    def submit_task():
        data = request.json
        global background_thread
        global skuid_backgroundthread
        logging.info(f'Received task: {data["task"]}')
        skuid = data["skuid"]

        if data["task"] == "start" and skuid not in skuid_backgroundthread:
            #if background_thread is None or not background_thread.is_alive():
            background_thread = BackgroundThreadFactory.create('handDetection')     
            skuid_backgroundthread[skuid] = background_thread
            background_thread.start()
            TASKS_QUEUE.put(data)
            print(skuid_backgroundthread) # Error checking
            return jsonify({'success': 'Thread Stared'}), 200
        elif data["task"] == "start" and skuid in skuid_backgroundthread:
            return jsonify({'status': 'Thread already running'}), 409
        
        if data["task"] == "stop" and skuid in skuid_backgroundthread:
            #if background_thread and background_thread.is_alive():
            background_thread = skuid_backgroundthread[skuid]
            background_thread.stop()
            background_thread.join()
            del skuid_backgroundthread[skuid]
            print(skuid_backgroundthread) # Error checking
            background_thread = None 
            return jsonify({'success': 'Thread stopped'}), 200
        elif data["task"] == "stop" and skuid not in skuid_backgroundthread:
            return jsonify({'status': 'No Thread running'}), 404
                

    return app

if __name__ == "__main__":
    app = create_app()
    app.run()
