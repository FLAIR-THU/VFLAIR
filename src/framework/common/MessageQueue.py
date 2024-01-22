import queue


class MessageQueue(object):
    _queue = queue.Queue(0)

    def put(self, data, block=True):
        self._queue.put(data)

    def get(self, block=True):
        return self._queue.get(block)


message_queue = MessageQueue()
