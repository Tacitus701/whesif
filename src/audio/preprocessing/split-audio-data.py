import math
import os

from pydub import AudioSegment


class SplitWavAudioMubin:
    def __init__(self, filepath, export_path, export_name):
        self.filepath = filepath
        self.export_path = export_path
        self.export_name = export_name

        self.audio = AudioSegment.from_wav(self.filepath).set_channels(1).set_frame_rate(22050)

    def get_duration(self):
        return self.audio.duration_seconds

    def save_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(os.path.join(self.export_path, split_filename), format="wav")

    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = self.export_name + "_" + str(i) + ".wav"
            self.save_split(i, i + min_per_split, split_fn)
            print(str(i) + " Done")
            if i == total_mins - min_per_split:
                print("All split successfully")


def main():
    filepath = "data/audio/macron.wav"
    export_path = "data/audio/split"
    export_name = "macron_split"
    split_wav = SplitWavAudioMubin(filepath, export_path, export_name)
    split_wav.multiple_split(min_per_split=1)


if __name__ == "__main__":
    main()
