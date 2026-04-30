from gnuradio import gr
import pmt
import numpy as np
import os


class binary_relseek_source(gr.sync_block):
    """
    GNURadio source block that reads raw interleaved float32 I/Q binary files.

    File format: pairs of float32 values [I0, Q0, I1, Q1, ...] with no header.
    One complex sample = 8 bytes (4 bytes I + 4 bytes Q).

    Parameters
    ----------
    file_path  : str   - Path to the raw binary I/Q file.
    sample_rate: float - Sample rate in Hz (used only for time-based seeks).
    start_sec  : float - Seek to this offset (seconds from start) on init.
    """

    BYTES_PER_SAMPLE = 8  # 2 x float32

    def __init__(self, file_path="", sample_rate=1e6, start_sec=0.0):
        gr.sync_block.__init__(
            self,
            name="binary_relseek_source",
            in_sig=None,
            out_sig=[np.complex64],
        )

        self.file_path = file_path
        self.fs = float(sample_rate)

        self._fh = None          # file handle
        self.total_samples = 0   # total samples in file
        self.cursor = 0          # current sample index

        self.message_port_register_in(pmt.intern("seek"))
        self.set_msg_handler(pmt.intern("seek"), self.handle_seek)

        if not self.file_path:
            print("[binary_relseek_source] Set file_path in block properties")
            return

        self._open_file()
        self.seek_seconds(float(start_sec))

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _open_file(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(
                f"[binary_relseek_source] File not found: '{self.file_path}'"
            )

        file_bytes = os.path.getsize(self.file_path)
        if file_bytes % self.BYTES_PER_SAMPLE != 0:
            print(
                f"[binary_relseek_source] WARNING: file size {file_bytes} bytes is not a "
                f"multiple of {self.BYTES_PER_SAMPLE}. Trailing bytes will be ignored."
            )

        self.total_samples = file_bytes // self.BYTES_PER_SAMPLE
        self._fh = open(self.file_path, "rb")
        self.cursor = 0

        print(
            f"[binary_relseek_source] Opened '{self.file_path}', "
            f"fs={self.fs} Hz, total_samples={self.total_samples} "
            f"({self.total_samples / self.fs:.3f} s)"
        )

    def stop(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return True

    # ------------------------------------------------------------------
    # Seek helpers
    # ------------------------------------------------------------------

    def _clamp(self, sample_idx: int) -> int:
        return max(0, min(sample_idx, self.total_samples - 1))

    def seconds_to_sample(self, seconds_from_start: float) -> int:
        return int(np.floor(seconds_from_start * self.fs))

    def seek_seconds(self, seconds_from_start: float):
        sample = self._clamp(self.seconds_to_sample(seconds_from_start))
        self._seek_to_sample(sample)
        print(
            f"[binary_relseek_source] Seek to {seconds_from_start:.6f}s -> sample {sample}"
        )

    def seek_samples_relative(self, sample_offset: int):
        sample = self._clamp(int(sample_offset))
        self._seek_to_sample(sample)
        print(
            f"[binary_relseek_source] Seek to sample_offset {sample_offset} -> sample {sample}"
        )

    def _seek_to_sample(self, sample_idx: int):
        self.cursor = sample_idx
        if self._fh is not None:
            self._fh.seek(sample_idx * self.BYTES_PER_SAMPLE)

    # ------------------------------------------------------------------
    # Message handler  (mirrors the original block exactly)
    # ------------------------------------------------------------------

    def handle_seek(self, msg):
        try:
            if self._fh is None:
                print("[binary_relseek_source] Ignoring seek: file not open")
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
                    self.seek_seconds(
                        float(pmt.to_double(pmt.dict_ref(msg, k_seconds, pmt.PMT_NIL)))
                    )
                    return
                if pmt.dict_has_key(msg, k_sample):
                    self.seek_samples_relative(
                        int(pmt.to_long(pmt.dict_ref(msg, k_sample, pmt.PMT_NIL)))
                    )
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
            print("[binary_relseek_source] Seek error:", e)

    # ------------------------------------------------------------------
    # Work
    # ------------------------------------------------------------------

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        if self._fh is None:
            out[:] = 0
            return n

        # Past EOF → zeros
        if self.cursor >= self.total_samples:
            out[:] = 0
            return n

        nreq = min(n, self.total_samples - self.cursor)

        try:
            raw = self._fh.read(nreq * self.BYTES_PER_SAMPLE)
            floats = np.frombuffer(raw, dtype=np.float32)

            # Pair up [I, Q, I, Q, ...] → complex64
            data = floats[0::2] + 1j * floats[1::2]
            data = data.astype(np.complex64)
            got = len(data)

            out[:got] = data
            if got < n:
                out[got:] = 0

            self.cursor += got

        except Exception as e:
            print("[binary_relseek_source] Read error:", e)
            out[:] = 0
            self.cursor = min(self.cursor + 1, self.total_samples)

        return n
