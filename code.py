from gnuradio import gr
import pmt
import numpy as np
import digital_rf

class digital_rf_seekable_source(gr.sync_block):
    def __init__(self, data_dir="", channel="", start_sample=0):
        gr.sync_block.__init__(
            self,
            name="digital_rf_seekable_source",
            in_sig=None,
            out_sig=[np.complex64],
        )

        self.data_dir = data_dir
        self.channel = channel
        self.cursor = int(start_sample)

        if not self.data_dir or not self.channel:
            # Let the block load in GRC even if params not set yet
            self.reader = None
            print("[digital_rf_seekable_source] Set data_dir and channel in block properties")
        else:
            self.reader = digital_rf.DigitalRFReader(self.data_dir)

        self.message_port_register_in(pmt.intern("seek"))
        self.set_msg_handler(pmt.intern("seek"), self.handle_seek)

    def handle_seek(self, msg):
        """
        Accepts:
          - PMT integer/float
          - PMT dict with key "sample"
          - PMT pair where car is command and cdr is sample
        """
        try:
            # Case 1: dict {"sample": N}
            if pmt.is_dict(msg):
                key = pmt.intern("sample")
                if pmt.dict_has_key(msg, key):
                    new_sample = int(pmt.to_long(pmt.dict_ref(msg, key, pmt.PMT_NIL)))
                else:
                    raise ValueError("seek dict missing 'sample' key")

            # Case 2: number (int/float)
            elif pmt.is_integer(msg):
                new_sample = int(pmt.to_long(msg))
            elif pmt.is_real(msg):
                new_sample = int(pmt.to_double(msg))

            # Case 3: pair ("seek" . N) or (anything . N)
            elif pmt.is_pair(msg):
                new_sample = int(pmt.to_long(pmt.cdr(msg)))

            else:
                # Fallback: try python conversion
                new_sample = int(pmt.to_python(msg))

            if new_sample < 0:
                new_sample = 0

            self.cursor = new_sample
            print(f"[digital_rf_seekable_source] Seek to sample {self.cursor}")

        except Exception as e:
            print("[digital_rf_seekable_source] Seek error:", e)

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)

        if self.reader is None:
            out[:] = 0
            return n

        try:
            data = self.reader.read_vector(self.cursor, n, self.channel)
            data = np.asarray(data, dtype=np.complex64)

            got = len(data)
            out[:got] = data
            if got < n:
                out[got:] = 0

            self.cursor += n

        except Exception as e:
            print("[digital_rf_seekable_source] Read error:", e)
            out[:] = 0

        return n