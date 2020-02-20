import imageio
import numpy as np
import matplotlib.pyplot as plt


def extract_grayscale_frames(video, start_time=0, data_freq=1):
    images = imageio.get_reader(video)
    fps = images.get_meta_data()['fps']

    # Extract the images from the video who adhere to starting after
    # "start_time" and occur at frequency "data_freq"
    # TODO:// When there is a stable release of all packages for 3.8 this can
    # be cleaned up to define the t := frame_num / fps - start_time
    temp = [((frame_num / fps - start_time), frame_num)
            for frame_num, _ in enumerate(images)
            if (frame_num / fps - start_time) >= 0
            and np.floor(np.round(((frame_num / fps - start_time) % data_freq)
                         * fps, decimals=10)) / fps == 0]

    times, frames_to_keep = list(zip(*temp))

    # Set the conversion from RGB to grayscale using the scikit-image method
    # (RGB to grayscale page)
    conversion = np.array([0.2125, 0.7154, 0.0721])
    grayscale_images = [g_im / g_im.max() for g_im in
                        [np.dot(im, conversion) for f, im in enumerate(images)
                         if f in frames_to_keep]]

    return times, grayscale_images


def output_plots(time, angles, base_width, volume, block=False):
    fig, ax1 = plt.subplots(figsize=(5, 5))
    color = 'black'
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Contact Angle [deg]', fontsize=10, color=color)
    ax1.plot(time, angles, marker='.', markerfacecolor=color,
             markeredgecolor=color, markersize=10, linestyle=None)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'red'
    ax2.set_ylabel('Baseline width [-]', fontsize=10, color=color)
    ax2.plot(time, base_width, marker='.', markerfacecolor=color,
             markeredgecolor=color, markersize=10, linestyle=None)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()
    if not block:
        plt.draw()
        plt.pause(2)
    else:
        plt.show()


def output_datafile(image, time, angles, base_width, volumes):

    if '\\' in image:
        parts = image.split('\\')
    else:
        parts = image.split('/')
    path = '/'.join(parts[:-1])  # Leave off the actual file part
    filename = path + f'/results_{parts[-1]}.csv'

    print(f'Saving the data to {filename}')

    with open(filename, 'w+') as file:
        file.write(f'Time,{",".join([str(t) for t in time])}')
        file.write('\n')
        file.write(f'Left angle,{",".join([str(s[0]) for s in angles])}')
        file.write('\n')
        file.write(f'Right angle,{",".join([str(s[1]) for s in angles])}')
        file.write('\n')
        file.write(f'Average angle,'
                   f'{",".join([str((s[1]+s[0])/2) for s in angles])}')
        file.write('\n')
        file.write(f'Baseline width,{",".join([str(s) for s in base_width])}')
        file.write('\n')
        file.write(f'Volume,{",".join([str(s) for s in volumes])}')
