from queues import RingBuf


class TestRingBuff:
    def setup(self):
        self.buffer = RingBuf(1000)

    def test_assert_memory_cycle_correctly(self):
        for i in range(2000):
            self.buffer.append(i)

        for i in range(1000):
            assert self.buffer[i] == i+1000