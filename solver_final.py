import cv2
import numpy as np
import math as m
from random import *
import matplotlib.pyplot as plt
import os
import argparse
import time

file_dir = os.path.dirname(os.path.realpath(__file__))

# Images of each possible shape, for finding the most similar shape on a given card
train_dir = os.path.join(file_dir, "train")
train_img_names = ["diamond", "oval", "squiggle"]
train_imgs = dict()

for img_name in train_img_names:
    train_img = cv2.imread(os.path.join(train_dir, f"{img_name}.png"))
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
    train_img = train_img.astype(np.uint8) * 6
    train_imgs[img_name] = train_img

# taken from assignment c1
def dilate(mask, morph_size=3, iterations=1):
    morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))

    mask = cv2.dilate(mask, morph_kern, iterations=iterations)
    return mask


# taken from assignment c1
def erode(mask, morph_size=3, iterations=1):
    morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))

    mask = cv2.erode(mask, morph_kern, iterations=iterations)
    return mask


def white_balance(card):
    # Adjusts the color so that the background is (nearly) white,
    # to adjust for different lighting conditions.
    card = card.copy()

    average_pixel = card[0:50] / 2 + card[450:500] / 2
    average_pixel = np.average(average_pixel, axis=0)
    average_pixel = np.average(average_pixel, axis=0)

    # correction to make the average background pixel totally white
    correction = [255, 255, 255] - average_pixel

    card = card.astype(np.uint16)
    correction = correction.astype(np.uint16)

    # get card height and width
    h, w, _ = card.shape

    correction = np.repeat([correction], w, axis=0)
    correction = np.repeat([correction], h, axis=0)

    card = card + correction
    card = np.clip(card, 0, 255)
    card = card.astype(np.uint8)
    return card


def get_card_color(card):
    # Gets the color of the card by converting to HSV, thresholding
    # based on saturation, and then finding the average hue to get the color.
    card_hsv = card.copy()
    card_hsv = cv2.cvtColor(card_hsv, cv2.COLOR_RGB2HSV)

    # card_saturated = card_hsv[:, :, 1] > np.average(card_hsv[:, :, 1]) * 2
    card_saturated = card_hsv[:, :, 1] > 50

    # get the average color of the card in saturated areas
    hue_pixels = card_hsv[card_saturated, 0]
    # Red-ish pixels are at both near 0 deg and 180 deg.
    # This boths both in the same region, preventing bugs like
    # the avearge of 0 and 180 being 90-ish (green or purple) instead of 0.
    hue_pixels[hue_pixels < 45] += 180

    if len(hue_pixels) == 0:
        return "??"

    hue = np.average(hue_pixels)

    if hue > 120 and hue < 160:
        return "purple"
    elif hue > 45 and hue < 80:
        return "green"
    elif hue > 160:
        return "red"
    else:
        return "??"


def get_cards(img):
    # Gets boxes around each card in the image, by thresholding a grayscale
    # image and finding contours.
    # From there, warps the pixels contained within each box to get a
    # flattened 300x500 image of the card.
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # threshold the image
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # find contours (polygons around each card)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get the 6th largest contour area
    median_card_area = cv2.contourArea(sorted_contours[5])

    # filter out contours that are too small or too large, i.e. contours that are unlikely to be a card
    contours = [
        c
        for c in sorted_contours
        if cv2.contourArea(c) > median_card_area * 0.5
        and cv2.contourArea(c) < median_card_area * 2
    ]

    boxes = []
    for c in contours:
        # approximate contour as a rectangle using approxPolyDP
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if approx.shape != (4, 1, 2):
            continue

        boxes.append(approx)

    # use each box to warp perspective
    warp_size = (300, 500)
    cards = []
    for box in boxes:
        pts = np.array(box, dtype="float32")
        pts = pts.reshape((4, 2))

        # get height and width of the box found
        width = np.sqrt((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2)
        height = np.sqrt((pts[0][0] - pts[3][0]) ** 2 + (pts[0][1] - pts[3][1]) ** 2)

        if width > height:
            # card is right-side up
            pts2 = np.float32(
                [
                    [0, 0],
                    [0, warp_size[1]],
                    [warp_size[0], warp_size[1]],
                    [warp_size[0], 0],
                ]
            )
            M = cv2.getPerspectiveTransform(pts, pts2)
            warped = cv2.warpPerspective(img, M, warp_size)

        else:
            # card is sideways
            pts2 = np.float32(
                [
                    [0, 0],
                    [0, warp_size[0]],
                    [warp_size[1], warp_size[0]],
                    [warp_size[1], 0],
                ]
            )
            M = cv2.getPerspectiveTransform(pts, pts2)
            warped = cv2.warpPerspective(img, M, (warp_size[1], warp_size[0]))
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        cards.append(warped)

    return cards, boxes


def edge_detect(card):
    # Detects edges using canny edge detection, and then
    # cleans it up with morphology.
    card = card.copy()

    # blur the card
    card = cv2.GaussianBlur(card, (11, 11), 0)

    # edge detection
    edges = cv2.Canny(card, 50, 100)

    edges_dilate = dilate(edges, iterations=5)

    return (edges, edges_dilate)


def get_blobs(edges):
    # Use opencv simple blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 3000
    params.maxArea = 25000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(edges)

    centers = [k.pt for k in keypoints]

    return (len(keypoints), centers, keypoints)


def get_avg_shape(filled_shapes, num_shapes):
    # Gets the average shape, given multiple filled in shapes
    # from the card image.

    # every pixel will be an int between 0 and 6
    # 6 is chosen because it's the LCM of the numbers 1-3
    avg_shape = np.zeros((112, 222), np.uint8)
    for shape in filled_shapes:
        avg_shape = avg_shape + shape * (6 / num_shapes)

    return avg_shape.astype(np.uint8)


def get_most_similar_shape(avg_shape):
    # Uses template matching to get the most similar shape
    min_dist = float("inf")
    min_shape = None
    for shape_name, shape in train_imgs.items():
        sub = cv2.absdiff(avg_shape, shape)
        dist = cv2.countNonZero(sub)
        if dist < min_dist:
            min_dist = dist
            min_shape = shape_name
    return min_shape


def get_shape_and_num_shapes(card):
    # Gets the shape and number of shapes on the card.

    # Find "blobs" of each shape on the card
    # the number of blobs == number of shapes on the card.
    _, edges_dilate = edge_detect(card)
    num_blobs, blob_centers, _ = get_blobs(edges_dilate)

    # for each blob, get a filled in shape
    filled_shapes = []

    for center in blob_centers:
        min_y = int(center[1]) - 55
        max_y = int(center[1]) + 55

        shape = edges_dilate[min_y:max_y, 40:260]
        h, w = shape.shape
        filled_shape = np.zeros((h + 2, w + 2), np.uint8)
        try:
            cv2.floodFill(shape, filled_shape, (110, 55), 255)
        except cv2.error:
            continue

        filled_shapes.append(filled_shape)

    # average the binary filled shape
    if num_blobs > 0:
        avg_shape = get_avg_shape(filled_shapes, num_blobs)

        # get the most similar shape
        shape_name = get_most_similar_shape(avg_shape)

        return (num_blobs, shape_name)
    else:
        return (0, "??")


def get_card_fill(card):
    # Get the fill (outline, striped, or solid) of the card.

    # Edge detect to get blobs for each shape
    card = card.copy()
    edges, edges_dilate = edge_detect(card)
    edges_erode = erode(edges_dilate, iterations=3)

    # flood fill the edges
    h, w = edges.shape
    filled_edges = np.zeros((h + 2, w + 2), np.uint8)

    # Create a mask, where the mask is 1 for pixels "inside" the card.
    mask = cv2.floodFill(edges_erode, filled_edges, (5, 5), 255)[1]
    mask = np.bitwise_not(mask)

    # Convert to hsv to get the saturation channel
    # Idea: High saturation is filled in, whereas low saturation
    # is mostly background pixels and so is outline fill. Stripe
    # is someplace in the middle.
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)

    s_mask = cv2.bitwise_and(s, mask)

    s_mask = s_mask.flatten()
    s_mask = s_mask[s_mask > 0]

    if len(s_mask) == 0:
        return "??"

    # get the average saturation
    avg_s = np.average(s_mask)

    if avg_s > 100:
        return "fill"
    elif avg_s > 10:
        return "stripe"
    else:
        return "outline"


def solve_sets(card_props):
    # Given a list of cards, brute forces all the possible sets
    sets = []
    for i in range(len(card_props)):
        for j in range(i + 1, len(card_props)):
            for k in range(j + 1, len(card_props)):
                could_be_set = True
                for p in range(4):
                    # check if any are equal to "??"
                    if (
                        card_props[i][p] == "??"
                        or card_props[j][p] == "??"
                        or card_props[k][p] == "??"
                    ):
                        could_be_set = False
                        break

                    # check if any 2 are equal
                    if (
                        card_props[i][p] == card_props[j][p]
                        or card_props[i][p] == card_props[k][p]
                        or card_props[j][p] == card_props[k][p]
                    ):
                        if (
                            card_props[i][p] != card_props[j][p]
                            or card_props[i][p] != card_props[k][p]
                        ):
                            could_be_set = False
                            break
                if could_be_set:
                    sets.append((i, j, k))
    return sets


def annotate_sets_on_frame(input_frame, debug=False):
    # Driver for the entire set detection pipeline.
    # Get the cards
    cards, boxes = get_cards(input_frame)

    # figure out the properties of each card
    card_props = []

    for i, card in enumerate(cards):
        card = white_balance(card)
        card_color = get_card_color(card)
        num_shapes, shape_name = get_shape_and_num_shapes(card)
        fill = get_card_fill(card)

        card_props.append((card_color, num_shapes, shape_name, fill))

    # solve for all the sets
    sets = solve_sets(card_props)

    # draw the sets on the frame
    output_img = input_frame.copy()

    if debug:
        # draw boxes around each card
        # output_img = cv2.drawContours(output_img, boxes, -1, (255, 0, 255), 5)

        # write the card properties
        for i, props in enumerate(card_props):
            (card_color, num_shapes, shape_name, fill) = props
            box = boxes[i]
            # text = f"{card_color}"
            # text = f"{i} {card_color} {num_shapes} {shape_name[0:4]} {fill[0:4]}"
            text = f"{card_color} {num_shapes} {shape_name[0:4]} {fill[0:4]}"
            cv2.putText(
                output_img,
                text,
                (box[0, 0][0], box[0, 0][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

    # annotate the sets
    set_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    for i, set in enumerate(sets[0:5]):
        for j in set:
            box = boxes[j]
            box += (10 * i, 10 * i)
            cv2.drawContours(output_img, [box], -1, set_colors[i], 10)

    return output_img


def process_video(
    input_fname, output_fname, disp_frames=True, write_video=True, read_every_n_frames=2
):
    # Given a video file, process it and output a video with annotated sets.
    cap = cv2.VideoCapture(input_fname)

    # get resolution and frame rate of video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if write_video:
        output_vid = cv2.VideoWriter(
            output_fname,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps / read_every_n_frames,
            (width, height),
        )

    frame_num = 0

    while frame_num < num_frames:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_frame = annotate_sets_on_frame(frame, debug=True)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            if disp_frames:
                cv2.imshow(f"Frame {frame_num}", annotated_frame)
            else:
                print(frame_num)
            if write_video:
                output_vid.write(annotated_frame)
        except Exception as e:
            print(f"Error: Skipping frame {frame_num}", e)

        frame_num += read_every_n_frames

    cap.release()

    if write_video:
        output_vid.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # get starting time
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Annotate sets on a video of cards")
    parser.add_argument(
        "--video",
        type=str,
        help="The path to the input video file",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="The path to the input image file",
    )
    parser.add_argument(
        "-o",
        type=str,
        help="The path to the output video file",
        default="./output.avi",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Whether to display the annotated frames",
    )
    args = parser.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_img = annotate_sets_on_frame(img, debug=True)
        plt.imshow(annotated_img)
        plt.show()
    else:
        process_video(
            args.video,
            args.o,
            disp_frames=args.display,
        )

    # ending time
    end_time = time.time()

    # print how long it took to run
    print(f"It took {end_time - start_time} seconds to run")
