import cv2
import numpy as np
from face_alignment_keras.utils import create_target_heatmap


def image_depth_generator(df, batch_size=32, size=(64, 64), shuffle=True):
    """
    Yields the next training batch.
    """
    num_samples = len(df.index)
    while True:
        idx = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(idx)

        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = idx[offset:offset + batch_size]

            # Initialise arrays for this batch
            batch_input = []
            batch_output = []

            # For each example
            for i in batch_samples:
                img_name = df.loc[i, 'img']
                img = cv2.imread(img_name)[..., ::-1]  # switch to rgb

                # crop and resize image
                # get the detected faces
                x0 = int(df.loc[i, 'x0'])
                x1 = int(df.loc[i, 'x1'])
                y0 = int(df.loc[i, 'y0'])
                y1 = int(df.loc[i, 'y1'])
                cropped_img = img[y0:y1, x0:x1, :]
                inp = cv2.resize(cropped_img, dsize=size, interpolation=cv2.INTER_LINEAR)

                # build heatmap
                cx = float(df.loc[i, 'cx'])
                cy = float(df.loc[i, 'cy'])
                scale = float(df.loc[i, 'scale'])
                center = [cx, cy]
                landmarks = df.loc[i].to_numpy()
                target_landmarks = landmarks[9:]  # keep only the landmarks pos
                target_landmarks = np.reshape(target_landmarks, (-1, 3)).astype(float)
                xy_landmarks = target_landmarks[:, 0:2].astype(int)
                # get depths labels
                z = target_landmarks[:, 2]

                scales = np.expand_dims(scale, 0)  # add the batch dim
                centers = np.expand_dims(center, 0)  # add the batch dim
                xy_landmarks = np.expand_dims(xy_landmarks, 0)
                heatmaps = create_target_heatmap(xy_landmarks, centers, scales)
                heatmaps = np.transpose(heatmaps[0], (1, 2, 0))

                # inp = preprocess_input(image=input)  # todo data augmentation
                inp = np.array(inp) / 255.0
                batch_input += [np.concatenate((inp, heatmaps), axis=2)]
                batch_output += [z]

            # Make sure they're numpy arrays (as opposed to lists)
            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)

            # The generator-y part: yield the next training batch
            yield batch_x, batch_y