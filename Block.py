from gnuradio import gr
import numpy as np
import digital_rf
import threading


class signal_relseek_source(gr.sync_block):
    def __init__(self,
                 source_type="digital_rf",   # "digital_rf" or "bin"
                 data_dir="",                # DigitalRF root OR bin file path
                 channel="",                 # only used for DigitalRF
                 seek_seconds=0.0,
                 subchannel=0,
                 bin_dtype="float32",        # for bin mode: float32, int16, complex64
                 sample_rate=522000):        # for bin mode

        gr.sync_block.__init__(
            self,
            name="signal_relseek_source",
            in_sig=None,
            out_sig=[np.complex64],
        )

        self.source_type = source_type
        self.data_dir = data_dir
        self.channel = channel
        self.subchannel = subchannel
        self.bin_dtype = bin_dtype
        self.sample_rate = float(sample_rate)

        self.reader = None
        self.bin_data = None

        self.fs = None
        self.start_sample = 0
        self.end_sample_excl = None
        self.cursor = 0

        self.lock = threading.Lock()
        self.pending_seek = None

        if not self.data_dir:
            print("[signal_relseek_source] Set data_dir / file path in block properties")
            return

        self._init_source()

        self.seek_seconds = float(seek_seconds)
        self.set_seek_seconds(self.seek_seconds)

    def _init_source(self):
        if self.source_type == "digital_rf":
            self._init_digital_rf()
        elif self.source_type == "bin":
            self._init_bin()
        else:
            raise ValueError(
                f"[signal_relseek_source] Unknown source_type '{self.source_type}'. "
                "Use 'digital_rf' or 'bin'."
            )

    def _init_digital_rf(self):
        self.reader = digital_rf.DigitalRFReader(self.data_dir)

        chans = self.reader.get_channels()
        if self.channel not in chans:
            raise ValueError(
                f"[signal_relseek_source] Channel '{self.channel}' not found. "
                f"Available: {chans}"
            )

        b0, b1 = self.reader.get_bounds(self.channel)
        self.start_sample = int(b0)
        self.end_sample_excl = int(b1)

        props = self.reader.get_properties(self.channel)
        if "samples_per_second" not in props:
            raise ValueError(
                "[signal_relseek_source] Missing 'samples_per_second' "
                f"in channel properties. Keys found: {list(props.keys())}"
            )

        self.fs = float(props["samples_per_second"])
        self.cursor = self.start_sample

        print(
            f"[signal_relseek_source] Opened DigitalRF data_dir='{self.data_dir}', "
            f"channel='{self.channel}', fs={self.fs} Hz, "
            f"bounds=[{self.start_sample}, {self.end_sample_excl})"
        )

    def _init_bin(self):
        dtype_map = {
            "float32": np.float32,
            "int16": np.int16,
            "complex64": np.complex64,
        }

        if self.bin_dtype not in dtype_map:
            raise ValueError(
                f"[signal_relseek_source] Unsupported bin_dtype '{self.bin_dtype}'. "
                "Use 'float32', 'int16', or 'complex64'."
            )

        raw_dtype = dtype_map[self.bin_dtype]
        raw = np.fromfile(self.data_dir, dtype=raw_dtype)

        if self.bin_dtype == "float32":
            if len(raw) % 2 != 0:
                raw = raw[:-1]
            self.bin_data = raw.view(np.complex64)

        elif self.bin_dtype == "int16":
            if len(raw) % 2 != 0:
                raw = raw[:-1]
            raw = raw.astype(np.float32)
            i = raw[0::2]
            q = raw[1::2]
            self.bin_data = (i + 1j * q).astype(np.complex64)

        elif self.bin_dtype == "complex64":
            self.bin_data = raw.astype(np.complex64)

        self.fs = float(self.sample_rate)
        self.start_sample = 0
        self.end_sample_excl = len(self.bin_data)
        self.cursor = 0

        print(
            f"[signal_relseek_source] Opened BIN file='{self.data_dir}', "
            f"bin_dtype='{self.bin_dtype}', fs={self.fs} Hz, "
            f"samples={self.end_sample_excl}"
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

            print(f"[signal_relseek_source] Requested seek -> {seek_seconds:.3f} s ({new_sample})")

        except Exception as e:
            print("[signal_relseek_source] Seek error:", e)

    def _read_digital_rf(self, start, nreq):
        data = self.reader.read_vector(start, nreq, self.channel)
        return np.asarray(data, dtype=np.complex64)

    def _read_bin(self, start, nreq):
        return self.bin_data[start:start + nreq]

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        if self.fs is None:
            out[:] = 0
            return n

        with self.lock:
            if self.pending_seek is not None:
                self.cursor = self.pending_seek
                self.pending_seek = None
                print(f"[signal_relseek_source] Seek applied -> sample {self.cursor}")

            local_cursor = self.cursor
            local_end = self.end_sample_excl

        if local_cursor >= local_end:
            out[:] = 0
            return n

        max_n = local_end - local_cursor
        nreq = min(n, max_n)

        try:
            if self.source_type == "digital_rf":
                data = self._read_digital_rf(local_cursor, nreq)
            else:
                data = self._read_bin(local_cursor, nreq)

            data = np.asarray(data, dtype=np.complex64)

            finite_mask = np.isfinite(data.real) & np.isfinite(data.imag)
            if not np.all(finite_mask):
                bad = np.count_nonzero(~finite_mask)
                print(f"[signal_relseek_source] Warning: replacing {bad} non-finite samples with 0")
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
            print(f"[signal_relseek_source] Read error at sample {local_cursor}: {e}")
            out[:] = 0

            with self.lock:
                if self.cursor == local_cursor:
                    self.cursor = min(self.cursor + nreq, self.end_sample_excl)

        return n
