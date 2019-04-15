BLACK = True
WHITE = False


def calculate_features(training_sample):    
    return [calculate_image_features(image) for image in training_sample]


def calculate_image_features(image):
    windows = [image[:, i] for i in range(len(image[0]))]
    features = [calculate_window_features(window) for window in windows]
    return features


def calculate_window_features(window):
    feature_functions = (upper_contour, lower_contour, b_w_transitions, number_of_black_pixels, fraction_of_black_pixels_between_uc_and_lc, fraction_of_black_pixels)
    return [feature_function(window) for feature_function in feature_functions]


def upper_contour(window):
    for i in range(len(window)):
        if window[i] == BLACK:
            return i
    return -1


def lower_contour(window):
    for i in range(len(window)-1, -1, -1):
        if window[i] == BLACK:
            return i
    return -1


def b_w_transitions(window):
    pixel = window[0]
    transitions = 0
    for i in window:
        if i != pixel:
            transitions += 1
            pixel = i
    return transitions


def number_of_black_pixels(window):
    number_of_black = 0
    for i in range(len(window)):
        if window[i] == BLACK:
            number_of_black += 1
    return number_of_black

def fraction_of_black_pixels(window):
    return number_of_black_pixels(window) / len(window) * 100

def fraction_of_black_pixels_between_uc_and_lc(window):
    uc = upper_contour(window)
    lc = lower_contour(window)
    if uc == lc or uc > lc:
        return 0.0
    black_pixels_between_uc_and_lc = number_of_black_pixels(window[uc:lc+1])
    return (black_pixels_between_uc_and_lc / ((lc+1)-uc)) * 100