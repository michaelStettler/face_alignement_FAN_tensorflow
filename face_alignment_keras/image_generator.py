import cv2
import numpy as np
from face_alignment_keras.utils import create_target_heatmap


def image_generator(df, epoch=100, batch_size=32, size=(256, 256), shuffle=True, custom_loss=False):
    """
    Yields the next training batch.
    """
    num_samples = len(df.index)
    is_3d = True
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
            ds = []

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

                # compute d
                d = np.sqrt((y1 - y0) * (x1 - x0))
                ds += [d]

                # build heatmap
                cx = float(df.loc[i, 'cx'])
                cy = float(df.loc[i, 'cy'])
                scale = float(df.loc[i, 'scale'])
                center = [cx, cy]
                landmarks = df.loc[i].to_numpy()
                target_landmarks = landmarks[9:]  # keep only the landmarks pos
                if is_3d:
                    target_landmarks = np.reshape(target_landmarks, (-1, 3)).astype(float)
                    target_landmarks = target_landmarks[:, 0:2].astype(int)
                else:
                    target_landmarks = np.reshape(target_landmarks, (-1, 2))

                scales = np.expand_dims(scale, 0)  # add the batch dim
                centers = np.expand_dims(center, 0)  # add the batch dim
                target_landmarks = np.expand_dims(target_landmarks, 0)
                heatmaps = create_target_heatmap(target_landmarks, centers, scales)

                # inp = preprocess_input(image=input)  # todo data augmentation
                inp = np.array(inp) / 255.0
                batch_input += [inp]
                batch_output += [np.transpose(heatmaps[0], (1, 2, 0))]

            # Make sure they're numpy arrays (as opposed to lists)
            batch_x = np.array(batch_input)
            batch_y = np.array(batch_output)
            ds = np.array(ds)

            # The generator-y part: yield the next training batch
            if custom_loss:
                yield {'inputs': batch_x, 'y_true': batch_y, 'd': ds}
            else:
                yield batch_x, batch_y


def get_batch(df, batch_size=32, size=(256, 256), shuffle=True):
    num_entries = len(df.index)
    print("num entries:", num_entries)
    batch_entries = np.arange(num_entries)
    if shuffle:
        batch_entries = np.random.choice(a=batch_entries, size=batch_size)
    print("batch_entries", batch_entries)

    is_3d = True
    batch_input = []
    batch_output = []
    ds = []
    # Read in each input, perform pre-processing and get labels
    for i in batch_entries:
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

        # compute d
        d = np.sqrt((y1 - y0) * (x1 - x0))
        ds += [d]

        # build heatmap
        cx = float(df.loc[i, 'cx'])
        cy = float(df.loc[i, 'cy'])
        scale = float(df.loc[i, 'scale'])
        center = [cx, cy]
        landmarks = df.loc[i].to_numpy()
        target_landmarks = landmarks[9:]  # keep only the landmarks pos
        if is_3d:
            target_landmarks = np.reshape(target_landmarks, (-1, 3)).astype(float)
            target_landmarks = target_landmarks[:, 0:2].astype(int)
        else:
            target_landmarks = np.reshape(target_landmarks, (-1, 2))

        scales = np.expand_dims(scale, 0)  # add the batch dim
        centers = np.expand_dims(center, 0)  # add the batch dim
        target_landmarks = np.expand_dims(target_landmarks, 0)
        heatmaps = create_target_heatmap(target_landmarks, centers, scales)

        # inp = preprocess_input(image=input)  # todo data augmentation
        inp = np.array(inp)/255.0
        batch_input += [inp]
        batch_output += [np.transpose(heatmaps[0], (1, 2, 0))]

    # Return a tuple of (input,output) to feed the network
    batch_x = np.array(batch_input)
    batch_y = np.array(batch_output)
    ds = np.array(ds)
    print("shape batch_x", np.shape(batch_x))
    print("shape batch_y", np.shape(batch_y))
    print("shape ds", np.shape(ds))

    return batch_x, batch_y, ds