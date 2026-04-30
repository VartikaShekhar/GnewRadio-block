from gnuradio import gr
import numpy as np
import digital_rf
import threading


class digital_rf_relseek_source(gr.sync_block):
    def __init__(self,
                 data_dir="",
                 channel="",
                 seek_seconds=0.0,
                 subchannel=0):

        gr.sync_block.__init__(
            self,
            name="digital_rf_relseek_source",
            in_sig=None,
            out_sig=[np.complex64],
        )

        self.data_dir = data_dir
        self.channel = channel
        self.subchannel = subchannel

        self.reader = None
        self.fs = None
        self.start_sample = 0
        self.end_sample_excl = None
        self.cursor = 0

        self.lock = threading.Lock()
        self.pending_seek = None

        if not self.data_dir or not self.channel:
            print("[digital_rf_relseek_source] Set data_dir and channel in block properties")
            return

        self._init_reader()
        self.seek_seconds = float(seek_seconds)
        self.set_seek_seconds(self.seek_seconds)

    def _init_reader(self):
        self.reader = digital_rf.DigitalRFReader(self.data_dir)

        chans = self.reader.get_channels()
        if self.channel not in chans:
            raise ValueError(
                f"[digital_rf_relseek_source] Channel '{self.channel}' not found. "
                f"Available: {chans}"
            )

        b0, b1 = self.reader.get_bounds(self.channel)
        self.start_sample = int(b0)
        self.end_sample_excl = int(b1)

        props = self.reader.get_properties(self.channel)
        if "samples_per_second" not in props:
            raise ValueError(
                "[digital_rf_relseek_source] Missing 'samples_per_second' "
                f"in channel properties. Keys found: {list(props.keys())}"
            )

        self.fs = float(props["samples_per_second"])
        self.cursor = self.start_sample

        print(
            f"[digital_rf_relseek_source] Opened data_dir='{self.data_dir}', "
            f"channel='{self.channel}', fs={self.fs} Hz, "
            f"bounds=[{self.start_sample}, {self.end_sample_excl})"
        )

    def seconds_to_sample(self, seconds_from_start: float) -> int:
        return int(self.start_sample + np.floor(seconds_from_start * self.fs))

    def sample_to_seconds(self, sample_idx: int) -> float:
        return float(sample_idx - self.start_sample) / float(self.fs)

    def clamp_sample(self, sample_idx: int) -> int:
        if sample_idx < self.start_sample:
            return self.start_sample
        last_valid = self.end_sample_excl - 1
        if sample_idx > last_valid:
            return last_valid
        return sample_idx

    def get_duration_s(self):
        if self.fs is None or self.end_sample_excl is None:
            return 0.0
        return float(self.end_sample_excl - self.start_sample) / float(self.fs)

    def get_current_second(self):
        with self.lock:
            if self.fs is None:
                return 0.0
            cur = self.cursor
        return self.sample_to_seconds(cur)

    def get_current_sample(self):
        with self.lock:
            return int(self.cursor)

    def set_seek_seconds(self, seek_seconds):
        try:
            seek_seconds = float(seek_seconds)
            new_sample = self.clamp_sample(self.seconds_to_sample(seek_seconds))

            with self.lock:
                self.pending_seek = new_sample

            print(f"[digital_rf_relseek_source] Requested seek -> {seek_seconds:.3f} s ({new_sample})")

        except Exception as e:
            print("[digital_rf_relseek_source] Seek error:", e)

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        if self.reader is None:
            out[:] = 0
            return n

        # short critical section
        with self.lock:
            if self.pending_seek is not None:
                self.cursor = self.pending_seek
                self.pending_seek = None
                print(f"[digital_rf_relseek_source] Seek applied -> sample {self.cursor}")

            local_cursor = self.cursor
            local_end = self.end_sample_excl

        if local_cursor >= local_end:
            out[:] = 0
            return n

        max_n = local_end - local_cursor
        nreq = min(n, max_n)

        try:
            data = self.reader.read_vector(
                local_cursor,
                nreq,
                self.channel
            )

            data = np.asarray(data, dtype=np.complex64)

            # sanitize NaN/Inf just in case
            finite_mask = np.isfinite(data.real) & np.isfinite(data.imag)
            if not np.all(finite_mask):
                bad = np.count_nonzero(~finite_mask)
                print(f"[digital_rf_relseek_source] Warning: replacing {bad} non-finite samples with 0")
                data = data.copy()
                data[~finite_mask] = 0.0 + 0.0j

            got = len(data)

            out[:got] = data
            if got < n:
                out[got:] = 0

            with self.lock:
                if self.cursor == local_cursor:
                    self.cursor += got

        except Exception as e:
            # likely a data gap or read issue
            print(f"[digital_rf_relseek_source] Read error at sample {local_cursor}: {e}")
            out[:] = 0

            # advance by the requested chunk so playback does not appear frozen
            with self.lock:
                if self.cursor == local_cursor:
                    self.cursor = min(self.cursor + nreq, self.end_sample_excl)

        return n
