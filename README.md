# Seekable Playback Source for GNU Radio

We built this tool to make working with recorded RF/binary data faster and more interactive. In most GNU Radio workflows, recordings are replayed linearly, making it hard to jump to specific time regions.

This project provides a **seekable playback interface**, allowing you to navigate large recordings using a slider instead of restarting or trimming data. With this, you can **scrub through RF recordings like a video timeline**.



## This project provides 2 main components



## 1. Seek Controller Script

The main entry point of this project is a standalone Python script that lets you interactively navigate DigitalRF recordings.

It builds and runs a GNU Radio flowgraph under the hood and adds a simple UI on top, so you can move through recorded RF data using a slider instead of replaying it linearly.

### How to Run

1. Make sure GNU Radio is installed and available in your environment.  
   If not, install it here: https://wiki.gnuradio.org/index.php/InstallingGR

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python src/seekable_source/seek_controller.py 
  --data-dir <PATH_TO_DIGITALRF_DATA> 
  --channel <CHANNEL_NAME> 
  --start-sec <START_TIME_SECONDS>
```

### Example

```bash
python src/seekable_source/seek_controller.py --data-dir mep12  --channel chA  --start-sec 400
```

---

## 2. GNU Radio Block

We also provide the underlying GNU Radio blocks used by the script.  
Use these if you want to build your own flowgraph instead of using the standalone script.

We provide two blocks:

- If using DigitalRF data, use `digital_rf_relseek_source`
- If using raw binary data, use `binary_relseek_source`



### Important

- The block does **not** include a slider UI. It exposes a seek parameter that can be connected to a variable.
- You must add your own control using GNU Radio blocks.
- Make sure GNU Radio Companion (GRC) is installed.



## How to Use in GNU Radio Companion

1. Open **GNU Radio Companion**
2. Add an **Embedded Python Block**
3. Paste the provided block code (`digital_rf_relseek_source` or `binary_relseek_source`)
4. Set required parameters:
   - data directory
   - channel
   - start time

5. Add a **QT Range** block. This acts as the slider.
6. Add a **Strobe** block. This pushes updates continuously.

7. Connect:
   - `QT Range → Strobe → Embedded Python Block` for seek control
   - `Embedded Python Block → DSP chain` for filters, decoders, and sinks

8. Add visualization blocks, such as QT GUI sinks, if needed.
9. Run the flowgraph.

---

## Example Flow

<img width="1001" height="483" alt="example flowgraph" src="https://github.com/user-attachments/assets/94ae1c25-00de-4a30-bc6a-1baa4e9ae630" />
