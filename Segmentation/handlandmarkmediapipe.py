import cv2
import mediapipe as mp
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
import numpy as np

def alpha_shape(points, alpha):
    if len(points) < 4:
        return Polygon(points)

    tri = Delaunay(points)
    triangles = points[tri.simplices]

    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circum_r = (a * b * c) / (4.0 * area)

    filtered = triangles[circum_r < 1.0 / alpha]
    edge_points = []

    for tri in filtered:
        for i in range(3):
            edge = (tuple(tri[i]), tuple(tri[(i+1)%3]))
            edge_points.append(edge)

    edge_line = MultiPoint([pt for edge in edge_points for pt in edge])
    m = polygonize(edge_line)
    return unary_union(list(m))

# Import the alpha_shape function from earlier
# (paste here or import from module)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            points = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append([x, y])
            points_np = np.array(points)

            # Compute alpha shape
            shape = alpha_shape(points_np, alpha=0.03)  # You can tune alpha here

            if isinstance(shape, Polygon):
                int_coords = np.array(shape.exterior.coords, dtype=np.int32)
                cv2.fillPoly(frame, [int_coords], (0, 255, 0))
                # or for outline only:
                # cv2.polylines(frame, [int_coords], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Concave Hull Hand", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
