import cv2
import mediapipe as mp
import warnings
import streamlit as st
from utils.feature_extraction import *
from utils.strings import *
from utils.model import ASLClassificationModel
from config import MODEL_NAME, MODEL_CONFIDENCE

# Ignore warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if __name__ == "__main__":
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam không hoạt động. Vui lòng kiểm tra lại!")
        exit()

    # Initialize ExpressionHandler
    try:
        expression_handler = ExpressionHandler()
    except NameError:
        st.error("Class `ExpressionHandler` chưa được định nghĩa!")
        exit()

    # Streamlit UI setup
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
            .big-font {
                color: #e76f51 !important;
                font-size: 60px !important;
                border: 0.5rem solid #fcbf49 !important;
                border-radius: 2rem;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([4, 2])
    video_placeholder = col1.empty()
    prediction_placeholder = col2.empty()

    # Load model
    try:
        model = ASLClassificationModel.load_model(f"models/model_mlp.pkl")
    except FileNotFoundError:
        st.error(f"Không tìm thấy mô hình: models/simple_expression_rf_model.pkl")
        exit()

    # Initialize MediaPipe Face Mesh and Hands
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MODEL_CONFIDENCE,
        min_tracking_confidence=MODEL_CONFIDENCE,
    )
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=MODEL_CONFIDENCE,
        min_tracking_confidence=MODEL_CONFIDENCE,
    )

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            st.warning("Không nhận được khung hình từ webcam.")
            continue
        
        image = cv2.flip(image, 1)
        # Convert image to RGB without flipping
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process face and hand landmarks
        face_results = face_mesh.process(image)
        hand_results = hands.process(image)

        # Extract features only if landmarks exist
        feature = []
        if face_results.multi_face_landmarks or hand_results.multi_hand_landmarks:
            feature = extract_features(mp_hands, face_results, hand_results)
            expression = model.predict(feature)
            expression_handler.receive(expression)

        # Draw landmarks
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                )

        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                )

        # Display video feed and prediction
        video_placeholder.image(image, channels="RGB", use_container_width=True)
        prediction_placeholder.markdown(
            f'''<h2 class="big-font">{expression_handler.get_message()}</h2>''', 
            unsafe_allow_html=True
        )

        # Exit if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
