from comfy_api.latest._util import VideoContainer, VideoCodec, VideoComponents
from fractions import Fraction
from typing import Optional
import av
import json

# patched up version of comfyui save video function
def video_save_to(
    self,
    buffer_or_path,
    format: VideoContainer = VideoContainer.AUTO,
    codec: VideoCodec = VideoCodec.AUTO,
    crf: int = 23,
    metadata: Optional[dict] = None
):
    if format != VideoContainer.AUTO and format != VideoContainer.MP4:
        raise ValueError("Only MP4 format is supported for now")
    if codec != VideoCodec.AUTO and codec != VideoCodec.H264:
        raise ValueError("Only H264 codec is supported for now")
    with av.open(buffer_or_path, mode='w', format="mp4", options={'movflags': 'use_metadata_tags'}) as output:
        # Add metadata before writing any streams
        if metadata is not None:
            for key, value in metadata.items():
                output.metadata[key] = json.dumps(value)

        self__components = self.get_components()

        frame_rate = Fraction(round(self__components.frame_rate * 1000), 1000)
        # Create a video stream
        video_stream = output.add_stream('h264', rate=frame_rate)
        video_stream.width = self__components.images.shape[2]
        video_stream.height = self__components.images.shape[1]
        video_stream.pix_fmt = 'yuv420p'
        video_stream.options = {'crf': str(crf)}

        # Create an audio stream
        audio_sample_rate = 1
        audio_stream: Optional[av.AudioStream] = None
        if self__components.audio:
            audio_sample_rate = int(self__components.audio['sample_rate'])
            audio_stream = output.add_stream('aac', rate=audio_sample_rate)
            audio_stream.sample_rate = audio_sample_rate
            audio_stream.format = 'fltp'

        # Encode video
        for i, frame in enumerate(self__components.images):
            img = (frame * 255).clamp(0, 255).byte().cpu().numpy() # shape: (H, W, 3)
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            frame = frame.reformat(format='yuv420p')  # Convert to YUV420P as required by h264
            packet = video_stream.encode(frame)
            output.mux(packet)

        # Flush video
        packet = video_stream.encode(None)
        output.mux(packet)

        if audio_stream and self__components.audio:
            # Encode audio
            samples_per_frame = int(audio_sample_rate / frame_rate)
            num_frames = self__components.audio['waveform'].shape[2] // samples_per_frame
            for i in range(num_frames):
                start = i * samples_per_frame
                end = start + samples_per_frame
                # TODO(Feature) - Add support for stereo audio
                chunk = (
                    self__components.audio["waveform"][0, 0, start:end]
                    .unsqueeze(0)
                    .contiguous()
                    .numpy()
                )
                audio_frame = av.AudioFrame.from_ndarray(chunk, format='fltp', layout='mono')
                audio_frame.sample_rate = audio_sample_rate
                audio_frame.pts = i * samples_per_frame
                for packet in audio_stream.encode(audio_frame):
                    output.mux(packet)

            # Flush audio
            for packet in audio_stream.encode(None):
                output.mux(packet)