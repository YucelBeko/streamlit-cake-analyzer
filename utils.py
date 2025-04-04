import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_13_zones(mask, center, max_radius):
    h, w = mask.shape
    cx, cy = center
    D2 = int(0.6 * max_radius)
    D3 = int(0.3 * max_radius)
    zones = []

    central_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(central_mask, (cx, cy), D3, 255, -1)
    zones.append(cv2.bitwise_and(central_mask, mask))

    for i in range(4):
        a1, a2 = i * 90, (i + 1) * 90
        mask_i = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask_i, (cx, cy), (D2, D2), 0, a1, a2, 255, -1)
        cv2.ellipse(mask_i, (cx, cy), (D3, D3), 0, a1, a2, 0, -1)
        zones.append(cv2.bitwise_and(mask_i, mask))

    for i in range(8):
        a1, a2 = i * 45, (i + 1) * 45
        mask_o = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask_o, (cx, cy), (max_radius, max_radius), 0, a1, a2, 255, -1)
        cv2.ellipse(mask_o, (cx, cy), (D2, D2), 0, a1, a2, 0, -1)
        zones.append(cv2.bitwise_and(mask_o, mask))

    return zones

def get_shade_number(ry_val):
    thresholds = [
        (7.2, 17), (9.3, 16), (12.2, 15), (16.4, 14), (20.1, 13),
        (22.9, 12), (26.5, 11), (31.7, 10), (38.5, 9), (46.9, 8),
        (54.2, 7), (64.3, 6), (75.2, 5)
    ]
    for limit, shade in thresholds:
        if ry_val < limit:
            return shade
    return 4

shade_color_map = {
    4:  (73, 74, 38),     5:  (45, 64, 54),     6:  (0, 255, 255),
    7:  (0, 192, 255),    8:  (128, 128, 255),  9:  (255, 0, 255),
    10: (128, 255, 255),  11: (0, 128, 128),    12: (255, 128, 128),
    13: (255, 0, 128),    14: (0, 255, 0),      15: (0, 128, 0),
    16: (255, 0, 0),      17: (128, 0, 0)
}

def analyze_cake_image(image_raw):
    output = image_raw.copy()
    mean_ry_values = []

    # Step 1: Brightness-adjusted masking
    enhanced = cv2.convertScaleAbs(image_raw, alpha=1.4, beta=35)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    non_white_mask = cv2.bitwise_not(white_mask)
    _, binary = cv2.threshold(non_white_mask, 1, 255, cv2.THRESH_BINARY)

    # Step 2: CLAHE Lab image
    lab = cv2.cvtColor(image_raw, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    image_clahe = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_Lab2BGR)

    # Step 3: Object detection
    num_labels, labels = cv2.connectedComponents(binary)

    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < 50:
            continue

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(cnts[0])
        cx, cy = x + w // 2, y + h // 2
        max_r = int(max(w, h) / 2)
        zones = get_13_zones(mask, (cx, cy), max_r)

        all_ry = []

        for zone_mask in zones:
            pixels = image_clahe[zone_mask > 0]
            if len(pixels) < 10:
                continue

            sorted_idx = np.argsort(np.mean(pixels, axis=1))
            pixels = pixels[sorted_idx[int(0.05 * len(sorted_idx)):int(0.95 * len(sorted_idx))]]
            avg_rgb = np.mean(pixels, axis=0)
            lab_val = cv2.cvtColor(np.uint8([[avg_rgb]]), cv2.COLOR_RGB2Lab)[0][0]
            ry = (lab_val[0] / 255.0) * 100
            shade = get_shade_number(ry)
            all_ry.append(ry)

            color = shade_color_map.get(shade, (80, 80, 80))
            mask_3c = cv2.merge([zone_mask]*3)
            fill = np.full_like(output, color)
            blended = cv2.addWeighted(output, 1.0, fill, 0.5, 0)
            output = np.where(mask_3c == 255, blended, output)

        if all_ry:
            mean_ry = np.mean(all_ry)
            mean_shade = get_shade_number(mean_ry)
            mean_ry_values.append(mean_ry)

            label_text = f"{mean_shade}"
            color_box = shade_color_map.get(mean_shade, (128, 128, 128))
            y_label = y - 25 if y > 25 else y + h + 20
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(output, (x, y_label - th - 4), (x + tw + 8, y_label), color_box, -1)
            cv2.putText(output, label_text, (x + 4, y_label - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Bar plot
    fig1 = plt.figure(figsize=(12, 4))
    plt.bar(range(len(mean_ry_values)), mean_ry_values, color='gray')
    plt.axhline(38.5, color='blue', linestyle='--', label='Shade 9')
    plt.axhline(46.9, color='red', linestyle='--', label='Shade 8')
    plt.axhline(42.7, color='green', linestyle='--', label='Hedef Ry')
    plt.xticks(range(len(mean_ry_values)), [f'Obj {i+1}' for i in range(len(mean_ry_values))], rotation=90)
    plt.ylabel("Ry")
    plt.title("Ortalama Ry Değerleri")
    plt.legend()
    plt.tight_layout()

    # Heatmap
    rows = int(np.sqrt(len(mean_ry_values)))
    cols = int(np.ceil(len(mean_ry_values) / rows))
    padded = mean_ry_values + [np.nan] * (rows*cols - len(mean_ry_values))
    data = np.array(padded).reshape((rows, cols))

    fig2 = plt.figure(figsize=(6, 4))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrBr", cbar_kws={'label': 'Ry'})
    plt.title("Isı Haritası: Ry")
    plt.tight_layout()

    return output, mean_ry_values, fig1, fig2
