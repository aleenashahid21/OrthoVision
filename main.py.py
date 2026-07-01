import traceback
try:

    import sys, os, cv2, warnings, math, hashlib
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")       
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from PIL import Image
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras import Input
    from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection   import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.preprocessing     import LabelEncoder, StandardScaler
    from sklearn.metrics           import (confusion_matrix, classification_report,
                                           ConfusionMatrixDisplay, accuracy_score,
                                           precision_score, recall_score, f1_score)
    from sklearn.utils.class_weight import compute_class_weight
    import warnings
    warnings.filterwarnings("ignore")
    sys.excepthook = sys.__excepthook__

    #config
    DATASET_PATH = r"C:\Users\aleen\Desktop\dataset"
    OUTPUT_DIR   = r"C:\Users\aleen\Desktop\OrthoVision\output"
    MODEL_PATH   = "pose_landmarker_full.task"
    IMG_SIZE     = (128, 128)
    BATCH_SIZE   = 16          # smaller batch as small dataset
    EPOCHS_CNN   = 30
    SEED         = 42
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tf.random.set_seed(SEED);  np.random.seed(SEED)

    #mediapipe
    BaseOptions      = mp.tasks.BaseOptions
    VisionRunningMode = vision.RunningMode
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    #feature extraction
    def get_pose_features(img_path):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return None
        h, w = img_bgr.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_bgr)
        result   = pose_landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        ls, rs   = pt(11), pt(12)
        lh, rh   = pt(23), pt(24)
        mid_sh   = (ls + rs) / 2
        mid_hp   = (lh + rh) / 2

        shoulder_tilt = abs(ls[1] - rs[1]) / h
        hip_tilt      = abs(lh[1] - rh[1]) / h
        spine_vec     = mid_hp - mid_sh
        spine_len     = np.linalg.norm(spine_vec) + 1e-6
        lateral_dev   = abs(spine_vec[0]) / spine_len
        cobb_proxy    = math.degrees(math.asin(np.clip(lateral_dev, 0, 1)))
        shoulder_w    = np.linalg.norm(rs - ls)
        hip_w         = np.linalg.norm(rh - lh)
        sh_hip_ratio  = shoulder_w / (hip_w + 1e-6)
        left_trunk    = np.linalg.norm(ls - lh)
        right_trunk   = np.linalg.norm(rs - rh)
        trunk_asym    = abs(left_trunk - right_trunk) / (left_trunk + right_trunk + 1e-6)

        return {
            "shoulder_tilt": round(shoulder_tilt, 4),
            "hip_tilt"     : round(hip_tilt,      4),
            "lateral_dev"  : round(lateral_dev,   4),
            "cobb_proxy"   : round(cobb_proxy,     2),
            "sh_hip_ratio" : round(sh_hip_ratio,   4),
            "trunk_asym"   : round(trunk_asym,     4),
        }

    #phash for duplicates that even seem to be
    def phash(img_path, hash_size=8):
        """Returns a perceptual hash string (detects visually identical images
        even when filenames differ — fixes the main duplicate problem)."""
        try:
            img = Image.open(img_path).convert("L").resize(
                (hash_size, hash_size), Image.LANCZOS)
            arr  = np.array(img, dtype=float)
            mean = arr.mean()
            bits = (arr > mean).flatten()
            return "".join(["1" if b else "0" for b in bits])
        except Exception:
            return None

    def hamming(h1, h2):
        return sum(c1 != c2 for c1, c2 in zip(h1, h2))

    #dataset builder
    def build_dataframe(dataset_path):
        records = []
        categories = {"normal": 0, "scoliosis": 1}

        for cat, label in categories.items():
            folder = os.path.join(dataset_path, cat)
            if not os.path.isdir(folder):
                print(f" Folder not found: {folder}"); continue

            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                try:
                    with Image.open(fpath) as img:
                        w, h = img.size
                        arr  = np.array(img.convert("L"))
                        brightness = round(arr.mean(), 2)
                        contrast   = round(arr.std(),  2)
                except Exception:
                    continue

                ph = phash(fpath)
                pf = get_pose_features(fpath) or {
                    "shoulder_tilt": np.nan, "hip_tilt": np.nan,
                    "lateral_dev":   np.nan, "cobb_proxy": np.nan,
                    "sh_hip_ratio":  np.nan, "trunk_asym": np.nan,
                }
                records.append({
                    "filepath"  : fpath,
                    "filename"  : fname,
                    "phash"     : ph,
                    "label_name": cat,
                    "label"     : label,
                    "width"     : w,
                    "height"    : h,
                    "brightness": brightness,
                    "contrast"  : contrast,
                    **pf
                })

        return pd.DataFrame(records)

    print("Building DataFrame …")
    df = build_dataframe(DATASET_PATH)
    print(f"✅Raw shape: {df.shape}")

    
    before = len(df)
    #remove exact filename duplicates
    df = df.drop_duplicates(subset="filename").reset_index(drop=True)
    # Then remove near-duplicates by phash (Hamming distance < 5)
    seen_hashes = []
    keep_mask   = []
    for ph in df["phash"]:
        if ph is None:
            keep_mask.append(True); continue
        is_dup = any(hamming(ph, sh) < 5 for sh in seen_hashes)
        if is_dup:
            keep_mask.append(False)
        else:
            seen_hashes.append(ph)
            keep_mask.append(True)

    df = df[keep_mask].reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicates (filename + perceptual hash)")
    print(f"Remaining: {len(df)} images\n")

    
    df["label_name"] = df["label_name"].astype("category")
    df["label"]      = df["label"].astype(int)
    pose_cols = ["shoulder_tilt","hip_tilt","lateral_dev",
                 "cobb_proxy","sh_hip_ratio","trunk_asym"]
    print("Missing values:\n", df.isnull().sum())
    for col in pose_cols:
        df[col].fillna(df[col].median(), inplace=True)

    print("\n── Descriptive Statistics ──────────────────────────────")
    print(df[["brightness","contrast","cobb_proxy",
              "shoulder_tilt","hip_tilt","trunk_asym"]].describe().round(3))
    print("\nClass counts:\n", df["label_name"].value_counts())

    #severity labelling...folder..then percentile cobb
    
    scol_df = df[df["label"] == 1]["cobb_proxy"]
    p33 = scol_df.quantile(0.33)
    p66 = scol_df.quantile(0.66)
    print(f"\nScoliosis cobb_proxy percentiles — p33={p33:.3f}°  p66={p66:.3f}°")

    def cobb_to_severity(row):
        if row["label"] == 0:
            return "Normal"
        
        cp = row["cobb_proxy"]
        if cp < p33:
            return "Mild"
        elif cp < p66:
            return "Moderate"
        else:
            return "Severe"

    df["severity"] = df.apply(cobb_to_severity, axis=1)
    print("\nSeverity distribution:\n", df["severity"].value_counts())

    #output pics
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("OrthoVision — EDA Dashboard (Deduped)", fontsize=14, fontweight="bold")

    sns.countplot(x="label_name", data=df, palette="Set2", ax=axes[0,0])
    axes[0,0].set_title("Class Distribution (after dedup)")

    sns.histplot(data=df, x="brightness", hue="label_name",
                 kde=True, palette="Set2", ax=axes[0,1])
    axes[0,1].set_title("Brightness Distribution")

    sns.boxplot(x="label_name", y="contrast", data=df,
                palette="Set2", ax=axes[0,2])
    axes[0,2].set_title("Contrast by Class")

    sns.histplot(data=df, x="cobb_proxy", hue="label_name",
                 kde=True, palette="Set2", ax=axes[1,0])
    axes[1,0].set_title("Cobb-Angle Proxy (°)")

    sns.scatterplot(data=df, x="shoulder_tilt", y="hip_tilt",
                    hue="label_name", palette="Set2", ax=axes[1,1])
    axes[1,1].set_title("Shoulder Tilt vs Hip Tilt")

    feat_cols_all = pose_cols + ["brightness","contrast","label"]
    corr = df[feat_cols_all].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=axes[1,2], linewidths=.5)
    axes[1,2].set_title("Feature Correlation")

    plt.tight_layout()
    eda_path = os.path.join(OUTPUT_DIR, "eda_dashboard.png")
    plt.savefig(eda_path, dpi=150); plt.close()
    print(f"\n EDA dashboard saved → {eda_path}")

    # Severity distribution plot
    fig2, ax2 = plt.subplots(figsize=(7,4))
    sev_order = ["Normal","Mild","Moderate","Severe"]
    sev_counts = df["severity"].value_counts().reindex(sev_order, fill_value=0)
    sns.barplot(x=sev_counts.index, y=sev_counts.values,
                palette=["#2ecc71","#f1c40f","#e67e22","#e74c3c"], ax=ax2)
    for i, v in enumerate(sev_counts.values):
        ax2.text(i, v + 0.5, str(v), ha="center", fontweight="bold")
    ax2.set_title("Severity Distribution (after percentile labeling)")
    ax2.set_ylabel("Count")
    sev_path = os.path.join(OUTPUT_DIR, "severity_dist.png")
    plt.tight_layout(); plt.savefig(sev_path, dpi=150); plt.close()
    print(f" Severity distribution saved → {sev_path}")

    #cnn, heavy augmentation, cross-val eval
    #aug genertor
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="training", shuffle=True, seed=SEED
    )
    val_gen = val_datagen.flow_from_directory(
        DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="validation", shuffle=False, seed=SEED
    )

    class_weights_arr = compute_class_weight(
        "balanced", classes=np.unique(train_gen.classes), y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights_arr))
    print("\nClass weights:", class_weight_dict)

    #model1
    base_model = MobileNetV2(input_shape=(128,128,3),
                             include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs  = Input(shape=(128,128,3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.5)(x)
    x       = layers.Dense(64,  activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model1 = Model(inputs, outputs, name="OrthoVision_CNN")
    model1.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    model1.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                restore_best_weights=True),
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR,"model1_best.keras"),
            monitor="val_loss", save_best_only=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, min_lr=1e-7)
    ]

    # Phase 1: frozen base
    print("\n── Training CNN head (base frozen) ──────────────────────")
    history1 = model1.fit(
        train_gen, epochs=EPOCHS_CNN,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=cbs
    )

    # Phase 2: fine-tune top 40 layers
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model1.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    print("\n── Fine-tuning top 40 layers ─────────────────────────────")
    history1_ft = model1.fit(
        train_gen, epochs=15,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=cbs
    )

    #Training curves
    def save_history_plot(h, title_prefix, fname):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(h.history["accuracy"],     label="train")
        ax1.plot(h.history["val_accuracy"], label="val")
        ax1.set_title(f"{title_prefix} — Accuracy"); ax1.legend()
        ax2.plot(h.history["loss"],     label="train")
        ax2.plot(h.history["val_loss"], label="val")
        ax2.set_title(f"{title_prefix} — Loss"); ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150); plt.close()

    save_history_plot(history1,    "Model 1 — Head Training", "train_head.png")
    save_history_plot(history1_ft, "Model 1 — Fine-tuning",   "train_ft.png")

    print("\n══ MODEL 1 EVALUATION ══════════════════════════════════")
    val_gen.reset()
    y_pred_prob = model1.predict(val_gen).ravel()
    y_pred      = (y_pred_prob > 0.5).astype(int)
    y_true      = val_gen.classes

    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("\nClassification Report:\n",
          classification_report(y_true, y_pred,
                                target_names=["Normal","Scoliosis"],
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal","Scoliosis"])
    fig, ax = plt.subplots(figsize=(6,5))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("Model 1 — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"cm_model1.png"), dpi=150); plt.close()

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Model 1 — ROC Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"roc_model1.png"), dpi=150); plt.close()
    print(f"ROC-AUC: {roc_auc:.4f}")

    #severity classifier RF...Stratified K-Fold cross-validation 
    FEAT_COLS = ["cobb_proxy","shoulder_tilt","hip_tilt",
                 "lateral_dev","sh_hip_ratio","trunk_asym",
                 "brightness","contrast"]

    X  = df[FEAT_COLS].values
    le = LabelEncoder()
    y_sev = le.fit_transform(df["severity"])
    sev_names = le.classes_
    n_classes = len(sev_names)

    print(f"\nSeverity classes: {sev_names}  (n={n_classes})")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    if n_classes < 2:
        print("⚠ Only one severity class found — Model 2 skipped.")
        print("  This means all scoliosis images had nearly identical cobb_proxy.")
        print("  Check your dataset or lower the percentile thresholds.")
        rf = None
        m2_acc = m2_f1 = m2_prec = m2_rec = float("nan")
    else:
        #Use cross_val_predict as small datset
        n_splits = min(5, min(np.bincount(y_sev)))
        n_splits = max(n_splits, 2)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED
        )
        # Full cross-val predictions for unbiased evaluation
        y_pred_cv = cross_val_predict(rf, X_sc, y_sev, cv=cv)
        cv_scores = cross_val_score(rf, X_sc, y_sev,
                                    cv=cv, scoring="f1_macro")
        print(f"CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        rf.fit(X_sc, y_sev) # Train final model on all data

        print("\n══ MODEL 2 EVALUATION (cross-val predictions) ══════════")
        m2_acc  = accuracy_score(y_sev, y_pred_cv)
        m2_prec = precision_score(y_sev, y_pred_cv, average="macro", zero_division=0)
        m2_rec  = recall_score(y_sev, y_pred_cv,    average="macro", zero_division=0)
        m2_f1   = f1_score(y_sev, y_pred_cv,        average="macro", zero_division=0)
        print(f"Accuracy  : {m2_acc:.4f}")
        print(f"Precision : {m2_prec:.4f}")
        print(f"Recall    : {m2_rec:.4f}")
        print(f"F1 Score  : {m2_f1:.4f}")
        print("\nClassification Report:\n",
              classification_report(y_sev, y_pred_cv,
                                    target_names=sev_names, zero_division=0))

        cm2 = confusion_matrix(y_sev, y_pred_cv)
        disp2 = ConfusionMatrixDisplay(cm2, display_labels=sev_names)
        fig, ax = plt.subplots(figsize=(7,6))
        disp2.plot(cmap="Greens", ax=ax)
        ax.set_title("Model 2 — Severity Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,"cm_model2.png"), dpi=150); plt.close()

        # ── FIX 8: Feature importance plot ───────────────────────
        importances = rf.feature_importances_
        feat_df = pd.DataFrame({"Feature": FEAT_COLS,
                                 "Importance": importances})
        feat_df.sort_values("Importance", ascending=False, inplace=True)

        plt.figure(figsize=(8,5))
        sns.barplot(data=feat_df, x="Importance", y="Feature",
                    palette="rocket")
        plt.title("Random Forest — Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR,"feature_importance.png"),
                    dpi=150); plt.close()
        print(f"\nTop feature: {feat_df.iloc[0]['Feature']} "
              f"({feat_df.iloc[0]['Importance']:.4f})")


    print("\n══ MODEL COMPARISON ════════════════════════════════════")
    m1_acc  = accuracy_score(y_true, y_pred)
    m1_f1   = f1_score(y_true, y_pred, zero_division=0)
    m1_prec = precision_score(y_true, y_pred, zero_division=0)
    m1_rec  = recall_score(y_true, y_pred, zero_division=0)

    comparison = pd.DataFrame({
        "Model"    : ["Model 1 — MobileNetV2 CNN",
                      "Model 2 — Random Forest"],
        "Task"     : ["Binary (Normal/Scoliosis)",
                      f"{n_classes}-class Severity"],
        "Accuracy" : [m1_acc,  m2_acc  if rf else float("nan")],
        "F1 Score" : [m1_f1,   m2_f1   if rf else float("nan")],
        "Precision": [m1_prec, m2_prec if rf else float("nan")],
        "Recall"   : [m1_rec,  m2_rec  if rf else float("nan")],
        "ROC-AUC"  : [roc_auc, "N/A"],
    })
    print(comparison.to_string(index=False))


    RECOMMENDATIONS = {
        "Normal": [
            "✅ Spine appears normal.",
            "💪 Maintain good posture when sitting and standing.",
            "🧘 Regular core-strengthening exercises recommended.",
        ],
        "Mild": [
            "⚠️  Mild scoliosis detected.",
            "🩺 Schedule a physiotherapy evaluation.",
            "🧘 Recommended: yoga, swimming, back-stretching routines.",
            "📅 Monitor every 6 months with a healthcare provider.",
        ],
        "Moderate": [
            "⚠️  Moderate scoliosis detected.",
            "🩺 Consult an orthopaedic specialist soon.",
            "🩹 Bracing may be recommended by your doctor.",
            "🏊 Low-impact exercise (swimming) can help.",
            "❌ Avoid heavy lifting or high-impact sports.",
        ],
        "Severe": [
            "🚨 Severe scoliosis detected.",
            "🏥 Seek immediate orthopaedic consultation.",
            "⚕️  Surgical intervention may be required.",
            "🛑 Restrict strenuous physical activity until assessed.",
        ],
    }

    def annotate_image(img_bgr, landmarks, cobb_proxy, severity, binary_conf):
        h, w = img_bgr.shape[:2]
        out  = img_bgr.copy()

        def lm_pt(idx):
            lm = landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))

        ls, rs = lm_pt(11), lm_pt(12)
        lh, rh = lm_pt(23), lm_pt(24)
        mid_sh = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
        mid_hp = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)

        cv2.line(out, ls, rs,   (255, 100, 0),  3)
        cv2.line(out, lh, rh,   (0,  165, 255), 3)
        cv2.line(out, mid_sh, mid_hp, (0, 200, 0), 3)
        for pt in [ls, rs, lh, rh, mid_sh, mid_hp]:
            cv2.circle(out, pt, 6, (0,0,255), -1)

        sev_col = {"Normal":(0,200,0),"Mild":(0,220,255),
                   "Moderate":(0,140,255),"Severe":(0,0,255)}
        cv2.putText(out, f"Cobb ~{cobb_proxy:.1f}deg", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(out, f"Severity: {severity}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    sev_col.get(severity,(255,255,255)), 2)
        cv2.putText(out, f"Scoliosis conf: {binary_conf:.1%}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        return out

    # Find optimal CNN threshold using val set (not fixed 0.5) ──
    # Use the threshold that maximises F1 on the validation set
    from sklearn.metrics import f1_score as f1
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.02):
        preds_t = (y_pred_prob > t).astype(int)
        f1_t = f1(y_true, preds_t, zero_division=0)
        if f1_t >= best_f1:
            best_f1 = f1_t
            best_thresh = t
    print(f"\n✅ Optimal CNN threshold: {best_thresh:.2f}  (F1={best_f1:.4f})")
    CNN_THRESHOLD = best_thresh

    def run_orthvision(img_path, model1, rf, scaler, le, save_path=None):
        """
        Dual-signal detection:
          Signal A — CNN confidence (learned from images)
          Signal B — Pose asymmetry score (geometric, model-free)
        A case is flagged as scoliosis if EITHER signal is positive.
        This prevents the CNN from overriding clear geometric evidence.
        """
        # ── Step 1: CNN binary detection ─────────────────────────
        img_pil = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
        img_arr = np.array(img_pil) / 255.0
        img_in  = np.expand_dims(img_arr, 0)
        conf    = float(model1.predict(img_in, verbose=0)[0][0])
        cnn_positive = conf > CNN_THRESHOLD

        # ── Step 2: Pose features ─────────────────────────────────
        img_bgr = cv2.imread(img_path)
        pose_feats_raw = get_pose_features(img_path) or {
            k: 0.0 for k in ["shoulder_tilt","hip_tilt","lateral_dev",
                              "cobb_proxy","sh_hip_ratio","trunk_asym"]
        }

        # median of scol class
        scol_mask = df["label"] == 1
        sh_tilt_med  = df.loc[scol_mask, "shoulder_tilt"].median()
        hip_tilt_med = df.loc[scol_mask, "hip_tilt"].median()
        trunk_med    = df.loc[scol_mask, "trunk_asym"].median()
        lat_med      = df.loc[scol_mask, "lateral_dev"].median()

        pose_positive = (
            pose_feats_raw["shoulder_tilt"] >= sh_tilt_med * 0.7 or
            pose_feats_raw["hip_tilt"]      >= hip_tilt_med * 0.7 or
            pose_feats_raw["trunk_asym"]    >= trunk_med * 0.7 or
            pose_feats_raw["lateral_dev"]   >= lat_med * 0.7
        )

        # ── Dual-signal decision ──────────────────────────────────
        # Scoliosis if CNN OR pose says so; Normal only if BOTH say normal
        is_scoliosis = cnn_positive or pose_positive

        detection_method = []
        if cnn_positive:   detection_method.append("CNN")
        if pose_positive:  detection_method.append("Pose")
        if not detection_method: detection_method.append("None")

        # ── Step 3: Severity (Model 2) ────────────────────────────
        grey = np.array(Image.open(img_path).convert("L"))
        feat_v = np.array([[
            pose_feats_raw["cobb_proxy"],
            pose_feats_raw["shoulder_tilt"],
            pose_feats_raw["hip_tilt"],
            pose_feats_raw["lateral_dev"],
            pose_feats_raw["sh_hip_ratio"],
            pose_feats_raw["trunk_asym"],
            grey.mean(),
            grey.std(),
        ]])

        if not is_scoliosis:
            severity = "Normal"
        elif rf is not None and n_classes > 1:
            feat_sc  = scaler.transform(feat_v)
            sev_code = rf.predict(feat_sc)[0]
            severity = le.inverse_transform([sev_code])[0]
            # Override if RF says Normal but pose says scoliosis
            if severity == "Normal" and pose_positive:
                cp = pose_feats_raw["cobb_proxy"]
                if cp < p33:   severity = "Mild"
                elif cp < p66: severity = "Moderate"
                else:          severity = "Severe"
        else:
            cp = pose_feats_raw["cobb_proxy"]
            if cp < p33:   severity = "Mild"
            elif cp < p66: severity = "Moderate"
            else:          severity = "Severe"

        # ── Step 4: Annotate ──────────────────────────────────────
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_bgr)
        result   = pose_landmarker.detect(mp_image)
        if result.pose_landmarks:
            annotated = annotate_image(img_bgr, result.pose_landmarks[0],
                                       pose_feats_raw["cobb_proxy"],
                                       severity, conf)
        else:
            annotated = img_bgr.copy()

        # ── Step 5: Save side-by-side ─────────────────────────────
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img_rgb);  axes[0].set_title("Original");             axes[0].axis("off")
        axes[1].imshow(ann_rgb);  axes[1].set_title(f"OrthoVision — {severity}"); axes[1].axis("off")
        plt.tight_layout()
        sp = save_path or os.path.join(OUTPUT_DIR, "inference_output.png")
        plt.savefig(sp, dpi=150); plt.close()

        print("\n" + "="*56)
        print("  OrthoVision Report")
        print("="*56)
        print(f"  Detection   : {'Scoliosis' if is_scoliosis else 'Normal'}")
        print(f"  CNN conf    : {conf:.1%}  (threshold={CNN_THRESHOLD:.2f})")
        print(f"  Triggered by: {' + '.join(detection_method)}")
        print(f"  Cobb Proxy  : {pose_feats_raw['cobb_proxy']:.2f}°")
        print(f"  Shoulder tilt: {pose_feats_raw['shoulder_tilt']:.4f}  "
              f"(scoliosis median={sh_tilt_med:.4f})")
        print(f"  Hip tilt     : {pose_feats_raw['hip_tilt']:.4f}  "
              f"(scoliosis median={hip_tilt_med:.4f})")
        print(f"  Trunk Asym   : {pose_feats_raw['trunk_asym']:.4f}  "
              f"(scoliosis median={trunk_med:.4f})")
        print(f"  Severity    : {severity}")
        print("  Recommendations:")
        for rec in RECOMMENDATIONS.get(severity, RECOMMENDATIONS["Normal"]):
            print(f"    {rec}")
        print("="*56)
        return annotated, severity, RECOMMENDATIONS.get(severity, [])

    # ── FIX: Pick the most representative sample from each folder ─
    def pick_best_sample(folder, prefer_high_cobb=True):
        """
        Instead of blindly picking [0], score each file by its pose
        features and return the one most clearly in its class.
        """
        files   = [f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        if not files:
            return None
        scored = []
        for f in files[:30]:   # cap at 30 to keep it fast
            fpath = os.path.join(folder, f)
            pf = get_pose_features(fpath)
            if pf:
                score = pf["cobb_proxy"] + pf["shoulder_tilt"]*100 + pf["trunk_asym"]*100
                scored.append((score, fpath))
        if not scored:
            return os.path.join(folder, files[0])
        scored.sort(reverse=prefer_high_cobb)
        return scored[0][1]

    # ── Run inference on BEST scoliosis sample ────────────────────
    scol_folder = os.path.join(DATASET_PATH, "scoliosis")
    if os.path.isdir(scol_folder):
        SAMPLE_IMG = pick_best_sample(scol_folder, prefer_high_cobb=True)
        if SAMPLE_IMG:
            print(f"\n📸 Scoliosis sample selected: {os.path.basename(SAMPLE_IMG)}")
            annotated_img, detected_severity, recs = run_orthvision(
                SAMPLE_IMG, model1, rf, scaler, le,
                save_path=os.path.join(OUTPUT_DIR, "inference_scoliosis.png")
            )
            cv2.imwrite(os.path.join(OUTPUT_DIR,"annotated_output.jpg"),
                        annotated_img)

    # ── Run inference on BEST normal sample ───────────────────────
    norm_folder = os.path.join(DATASET_PATH, "normal")
    if os.path.isdir(norm_folder):
        SAMPLE_NORM = pick_best_sample(norm_folder, prefer_high_cobb=False)
        if SAMPLE_NORM:
            print(f"\n📸 Normal sample selected: {os.path.basename(SAMPLE_NORM)}")
            run_orthvision(
                SAMPLE_NORM, model1, rf, scaler, le,
                save_path=os.path.join(OUTPUT_DIR, "inference_normal.png")
            )

    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")
    print("   Files: eda_dashboard.png, severity_dist.png, train_head.png,")
    print("          train_ft.png, cm_model1.png, roc_model1.png,")
    print("          cm_model2.png, feature_importance.png,")
    print("          inference_scoliosis.png, inference_normal.png,")
    print("          annotated_output.jpg, model1_best.keras")
    print(f"  CNN threshold: {CNN_THRESHOLD:.2f} (auto-tuned on val set)")

    pass

except Exception as e:
    print("Caught exception:", e)
    traceback.print_exc()