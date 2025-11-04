# Fill these with your Telegram bot credentials
# You can also set them via environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_CHAT_ID

TELEGRAM_BOT_TOKEN = "8164979830:AAGMPFSvc-yGPfUTxIfFG2AV_70IlrQ9yCk"
TELEGRAM_ADMIN_CHAT_ID = "7361910235"  # e.g., 123456789

# Strictness toggle for recognition thresholds
STRICT_MODE = True

# Base thresholds (more lenient)
BASE_UNKNOWN_THRESHOLD = 0.92
BASE_MARGIN_MIN = 0.30
BASE_ABS_MIN_PROB = 0.70

# Strict thresholds (default)
STRICT_UNKNOWN_THRESHOLD = 0.92
STRICT_MARGIN_MIN = 0.30
STRICT_ABS_MIN_PROB = 0.70

# Export effective thresholds based on STRICT_MODE
if STRICT_MODE:
    UNKNOWN_THRESHOLD = STRICT_UNKNOWN_THRESHOLD
    MARGIN_MIN = STRICT_MARGIN_MIN
    ABS_MIN_PROB = STRICT_ABS_MIN_PROB
else:
    UNKNOWN_THRESHOLD = BASE_UNKNOWN_THRESHOLD
    MARGIN_MIN = BASE_MARGIN_MIN
    ABS_MIN_PROB = BASE_ABS_MIN_PROB

# Minimum seconds between two alerts to avoid spamming
ALERT_COOLDOWN_SECONDS = 15

# Length of the clip to record when an unknown is detected
UNKNOWN_CLIP_SECONDS = 5

# Output directory for saved clips
UNKNOWN_CLIPS_DIR = "output/unknown_clips"

# Camera target FPS (attempted; actual may depend on hardware/driver)
CAMERA_TARGET_FPS = 60

# Recording FPS for saved clips
CLIP_FPS = 60

# Resize width for processing (smaller is faster). Typical values: 480, 600, 720
FRAME_WIDTH = 640

# Only run embedding/classification every N frames (>=1). 2 or 3 speeds up a lot.
PROCESS_EVERY_N = 2

# Minimum confidence for face detector (0..1). Lower to 0.3 if faces are not detected.
FACE_DET_CONFIDENCE = 0.5

# Optional: Enhance face ROI before embedding to improve recognition on low-light/soft images
ENHANCE_FACE = True
# CLAHE controls
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
# Unsharp mask controls
UNSHARP_AMOUNT = 1.0  # 0=off, 1=moderate, 1.5=strong
UNSHARP_KERNEL = (0, 0)  # (0,0) lets GaussianBlur decide from sigma
UNSHARP_SIGMA = 1.0
# Expand detected bounding box by this fraction to include more context (e.g., 0.10 = 10%)
BBOX_EXPAND_RATIO = 0.10

# Additional guard for unknowns: require top class to beat second-best by this margin.
# If (top_prob - second_prob) < MARGIN_MIN, classify as Unknown.
# (value set above by STRICT_MODE)

# Absolute minimum probability to accept a prediction as known. Below this, always mark Unknown.
# (value set above by STRICT_MODE)
