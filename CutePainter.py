import cv2
import numpy as np
import time
import os
import glob
import HandTrackingModule as htm
from collections import deque

drawing_base = None  # Used to overwrite files in 'continue' mode

# === Extract contours from transparent overlay ===
def extract_contours_from_overlay(overlay_img):
    alpha = overlay_img[:, :, 3]
    gray = cv2.cvtColor(overlay_img[:, :, :3], cv2.COLOR_BGR2GRAY)
    _, binary_lines = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(binary_lines, binary_lines, mask=alpha)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 100000]

# === Function to create a transparent mandala ===
def create_transparent_mandala(input_path, output_path):
    img = cv2.imread(input_path)
    img = cv2.resize(img, (1280, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    alpha = cv2.GaussianBlur(binary_inv, (3, 3), 0)
    b, g, r = cv2.split(img)
    rgba = [b, g, r, alpha]
    final = cv2.merge(rgba)
    cv2.imwrite(output_path, final)
    print(f"\u2705 Transparent mandala saved as {output_path}")

# === Auto-convert mandalas ===
def batch_convert_mandalas():
    os.makedirs("Mandalas", exist_ok=True)
    for path in glob.glob("Mandalas/*.jpg") + glob.glob("Mandalas/*.jpeg") + glob.glob("Mandalas/*.png"):
        if "_transparent" not in path:
            name = os.path.splitext(os.path.basename(path))[0]
            out_path = f"Mandalas/{name}_transparent.png"
            if not os.path.exists(out_path):
                create_transparent_mandala(path, out_path)

batch_convert_mandalas()

# === Settings ===
brushThick = 15
eraserThick = 100
saveFramesRequired = 20
MAX_UNDO = 5

folderPath = "Header"
overLayList = [cv2.imread(f'{folderPath}/{img}') for img in os.listdir(folderPath)]
header = overLayList[0]
drawColor = (255, 105, 180)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetecor(detCon=0.85)

# === Mode Selection ===
def choose_mode():
    print("\n\U0001f3a8 Choose Drawing Mode:")
    print("1. Start new blank drawing")
    print("2. Color a mandala")
    print("3. Continue a saved drawing")

    choice = input("Enter your choice (1/2/3): ").strip()
    if choice == '1':
        return 'new', None
    elif choice == '2':
        mandalas = glob.glob("Mandalas/*_transparent.png")
        print("\nAvailable mandalas:")
        for i, m in enumerate(mandalas):
            print(f"{i + 1}. {os.path.basename(m)}")
        index = int(input("Choose a mandala: ")) - 1
        return 'mandala', mandalas[index] if 0 <= index < len(mandalas) else None
    elif choice == '3':
        previews = glob.glob("SavedDrawings/*_preview.png")
        print("\nAvailable saved drawings:")
        for i, p in enumerate(previews):
            print(f"{i + 1}. {os.path.basename(p)}")
        index = int(input("Choose a drawing: ")) - 1

        if 0 <= index < len(previews):
            preview_path = previews[index]
            base = preview_path.replace("_preview.png", "")
            paint_path = base + "_paint.png"
            overlay_path = base + "_overlay.png"
            global drawing_base
            drawing_base = base  # save for later overwrite
            return 'continue', (paint_path, overlay_path)
        else:
            return 'new', None
    else:
        print("Invalid choice. Starting with blank drawing.")
        return 'new', None

# === Initialize State ===
mode, selected_image_path = choose_mode()
undo_stack = deque(maxlen=MAX_UNDO)
redo_stack = deque(maxlen=MAX_UNDO)
fillMode = False
mandala_img_overlay = None
mandala_contours = []
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# === Load modes ===
if mode == 'mandala' and selected_image_path:
    mandala_img_overlay = cv2.imread(selected_image_path, cv2.IMREAD_UNCHANGED)
    mandala_img_overlay = cv2.resize(mandala_img_overlay, (1280, 720))
    mandala_contours = extract_contours_from_overlay(mandala_img_overlay)

elif mode == 'continue' and selected_image_path:
    paint_path, overlay_path = selected_image_path
    painted = cv2.imread(paint_path, cv2.IMREAD_UNCHANGED)
    if painted is None:
        print("\u274c Failed to load saved paint image!")
        exit()

    painted = cv2.resize(painted, (1280, 720))
    b, g, r, a = cv2.split(painted)
    imgCanvas = cv2.merge((b, g, r))
    imgCanvas[a == 0] = 0

    if os.path.exists(overlay_path):
        mandala_img_overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        mandala_img_overlay = cv2.resize(mandala_img_overlay, (1280, 720))
        mandala_contours = extract_contours_from_overlay(mandala_img_overlay)

# === Save Drawing ===
def save_drawing():
    os.makedirs("SavedDrawings", exist_ok=True)
    global drawing_base

    if mode == 'continue' and drawing_base:
        filename_base = drawing_base
    else:
        filename_base = f"SavedDrawings/Drawing_{int(time.time())}"

    # 1. Save paint layer (user drawing with alpha)
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(imgCanvas)
    drawing_rgba = cv2.merge((b, g, r, alpha))
    cv2.imwrite(f"{filename_base}_paint.png", drawing_rgba)

    # 2. Save overlay layer (if any)
    if mandala_img_overlay is not None:
        cv2.imwrite(f"{filename_base}_overlay.png", mandala_img_overlay)

    # 3. Create white preview image
    white = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    mask = alpha
    color_part = cv2.bitwise_and(imgCanvas, imgCanvas, mask=mask)
    bg_part = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))
    preview = cv2.add(bg_part, color_part)

    if mandala_img_overlay is not None:
        overlay_rgb = mandala_img_overlay[:, :, :3].astype(np.float32)
        alpha_overlay = mandala_img_overlay[:, :, 3] / 255.0
        for c in range(3):
            preview[:, :, c] = preview[:, :, c] * (1 - alpha_overlay) + overlay_rgb[:, :, c] * alpha_overlay

    cv2.imwrite(f"{filename_base}_preview.png", preview)
    print(f"\u2705 Drawing saved as {filename_base}_preview.png (and paint + overlay)")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        thumbX, thumbY = lmList[4][1:]
        indexX, indexY = lmList[8][1:]
        distance = np.hypot(indexX - thumbX, indexY - thumbY)
        brushThick = int(np.interp(distance, [30, 200], [5, 50]))
        eraserThick = int(np.interp(distance, [30, 200], [20, 100]))

        if fingers[0] == 1 and all(f == 0 for f in fingers[1:]):
            saveGestureCount += 1
            print(f"Thumbs-up detectat! {saveGestureCount}/{saveFramesRequired}")
            if saveGestureCount >= saveFramesRequired:
                save_drawing()
                saveGestureCount = 0
        else:
            saveGestureCount = 0

        if fingers[1] and fingers[2]:
            xp, yp = None, None
            print("Mod Selectare")
            if y1 < 125:
                if 175 < x1 < 279:
                    header = overLayList[0]
                    drawColor = (255, 105, 180)
                elif 369 < x1 < 462:
                    header = overLayList[1]
                    drawColor = (255, 0, 0)
                elif 560 < x1 < 655:
                    header = overLayList[2]
                    drawColor = (0, 0, 255)
                elif 760 < x1 < 845:
                    header = overLayList[3]
                    drawColor = (0, 100, 0)
                elif 880 < x1 < 975:
                    header = overLayList[4]
                    drawColor = (0, 255, 255)
                elif 1100 < x1 < 1200:
                    header = overLayList[5]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        elif fingers[1] and not fingers[2]:
            if fillMode and mandala_contours:
                for cnt in mandala_contours:
                    if cv2.pointPolygonTest(cnt, (x1, y1), False) >= 0:
                        undo_stack.append(imgCanvas.copy())
                        redo_stack.clear()
                        mask = np.zeros(imgCanvas.shape[:2], dtype=np.uint8)
                        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
                        for c in range(3):
                            imgCanvas[:, :, c] = np.where(mask == 255, drawColor[c], imgCanvas[:, :, c])
                        print("✅ Filled inside shape")
                        break
                else:
                    print("⚠️ Not inside any shape")
            else:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                print("Mod Desenare")
                if xp is None and yp is None:
                    xp, yp = x1, y1
                    undo_stack.append(imgCanvas.copy())
                    redo_stack.clear()
                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThick)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThick)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThick)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThick)
                xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    if mandala_img_overlay is not None:
        overlay_rgb = mandala_img_overlay[:, :, :3].astype(np.float32)
        alpha_mask = mandala_img_overlay[:, :, 3] / 255.0
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - alpha_mask) + overlay_rgb[:, :, c] * alpha_mask

    if fillMode:
        cv2.putText(img, "FILL MODE ON", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    img[0:125, 0:1280] = header
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('s'):
        save_drawing()
    elif key == ord('u') and undo_stack:
        redo_stack.append(imgCanvas.copy())
        imgCanvas = undo_stack.pop()
        print("↩️ Undo")
    elif key == ord('r') and redo_stack:
        undo_stack.append(imgCanvas.copy())
        imgCanvas = redo_stack.pop()
        print("➡️ Redo")
    elif key == ord('c'):
        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
        undo_stack.clear()
        redo_stack.clear()
        print("🪼 Canvas curățat")
    elif key == ord('f'):
        fillMode = not fillMode
        print(f"🧺 Fill Mode {'ON' if fillMode else 'OFF'}")
