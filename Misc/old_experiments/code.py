from gnuradio import gr
import pmt
import numpy as np
import digital_rf


class digital_rf_relseek_source(gr.sync_block):
    def __init__(self, data_dir="", channel="", start_sec=0.0, subchannel=0):
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
        self.end_sample_excl = None   # IMPORTANT: end is exclusive
        self.cursor = 0

        self.message_port_register_in(pmt.intern("seek"))
        self.set_msg_handler(pmt.intern("seek"), self.handle_seek)

        if not self.data_dir or not self.channel:
            print("[digital_rf_relseek_source] Set data_dir and channel in block properties")
            return

        self._init_reader()
        self.seek_seconds(float(start_sec))

    def _init_reader(self):
        self.reader = digital_rf.DigitalRFReader(self.data_dir)

        chans = self.reader.get_channels()
        if self.channel not in chans:
            raise ValueError(
                f"[digital_rf_relseek_source] Channel '{self.channel}' not found. Available: {chans}"
            )

        b0, b1 = self.reader.get_bounds(self.channel)
        self.start_sample = int(b0)
        self.end_sample_excl = int(b1)   # EXCLUSIVE END

        props = self.reader.get_properties(self.channel)
        if "samples_per_second" not in props:
            raise ValueError(
                "[digital_rf_relseek_source] Missing 'samples_per_second' in channel properties. "
                f"Keys found: {list(props.keys())}"
            )
        self.fs = float(props["samples_per_second"])

        self.cursor = self.start_sample

        print(
            f"[digital_rf_relseek_source] Opened data_dir='{self.data_dir}', channel='{self.channel}', "
            f"fs={self.fs} Hz, bounds=[{self.start_sample}, {self.end_sample_excl})"
        )

    def seconds_to_sample(self, seconds_from_start: float) -> int:
        return int(self.start_sample + np.floor(seconds_from_start * self.fs))

    def clamp_sample(self, sample_idx: int) -> int:
        # clamp into valid [start, end_excl-1]
        if sample_idx < self.start_sample:
            return self.start_sample
        last_valid = self.end_sample_excl - 1
        if sample_idx > last_valid:
            return last_valid
        return sample_idx

    def seek_seconds(self, seconds_from_start: float):
        self.cursor = self.clamp_sample(self.seconds_to_sample(seconds_from_start))
        print(f"[digital_rf_relseek_source] Seek to {seconds_from_start:.6f}s -> sample {self.cursor}")

    def seek_samples_relative(self, sample_offset: int):
        self.cursor = self.clamp_sample(self.start_sample + int(sample_offset))
        print(f"[digital_rf_relseek_source] Seek to sample_offset {sample_offset} -> sample {self.cursor}")

    def handle_seek(self, msg):
        try:
            if self.reader is None:
                print("[digital_rf_relseek_source] Ignoring seek: reader not initialized")
                return

            if pmt.is_real(msg):
                self.seek_seconds(float(pmt.to_double(msg)))
                return
            if pmt.is_integer(msg):
                self.seek_samples_relative(int(pmt.to_long(msg)))
                return

            if pmt.is_dict(msg):
                k_seconds = pmt.intern("seconds")
                k_sample  = pmt.intern("sample")

                if pmt.dict_has_key(msg, k_seconds):
                    self.seek_seconds(float(pmt.to_double(pmt.dict_ref(msg, k_seconds, pmt.PMT_NIL)))
		)
                    return
                if pmt.dict_has_key(msg, k_sample):
                    self.seek_samples_relative(int(pmt.to_long(pmt.dict_ref(msg, k_sample, pmt.PMT_NIL))))
                    return

                raise ValueError("seek dict must contain key 'seconds' or 'sample'")

            py = pmt.to_python(msg)
            if isinstance(py, float):
                self.seek_seconds(py)
            elif isinstance(py, int):
                self.seek_samples_relative(py)
            else:
                raise ValueError(f"unsupported seek message type: {type(py)}")

        except Exception as e:
            print("[digital_rf_relseek_source] Seek error:", e)

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        if self.reader is None:
            out[:] = 0
            return n

        # If we're at/past end, just output zeros
        if self.cursor >= self.end_sample_excl:
            out[:] = 0
            return n

        # NEVER ask for samples past end_excl
        max_n = self.end_sample_excl - self.cursor
        nreq = min(n, max_n)

        try:
            # Depending on your digital_rf version, you may need subchannel as 4th arg
            # data = self.reader.read_vector(self.cursor, nreq, self.channel, self.subchannel)
            data = self.reader.read_vector(self.cursor, nreq, self.channel)

            data = np.asarray(data, dtype=np.complex64)
            got = len(data)

            out[:got] = data
            if got < n:
                out[got:] = 0

            self.cursor += got  # safe now because we limited nreq

        except Exception as e:
            print("[digital_rf_relseek_source] Read error:", e)
            out[:] = 0
            # Optional: bump cursor by 1 to avoid infinite loop on same bad index
            self.cursor = min(self.cursor + 1, self.end_sample_excl)

        return n
