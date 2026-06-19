import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

# =====================
# 1. Data Load
# =====================
X_train = np.load('models/X_train.npy')
X_test  = np.load('models/X_test.npy')
y_train = np.load('models/y_train.npy')
y_test  = np.load('models/y_test.npy')
le      = pickle.load(open('models/label_encoder.pkl', 'rb'))

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# =====================
# 2. KNN
# =====================
print("\n⏳ Training KNN...")
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print(f"✅ KNN Accuracy: {knn_acc*100:.2f}%")

# =====================
# 3. Random Forest
# =====================
print("\n⏳ Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"✅ Random Forest Accuracy: {rf_acc*100:.2f}%")

# =====================
# 4. ANN
# =====================
print("\n⏳ Training ANN...")
ann = Sequential([
    Dense(128, activation='relu',
          kernel_regularizer=l2(0.001),
          input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu',
          kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(len(np.unique(y_train)), activation='softmax')
])

ann.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

history = ann.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

ann_loss, ann_acc = ann.evaluate(X_test, y_test, verbose=0)
ann_pred = np.argmax(ann.predict(X_test), axis=1)
print(f"✅ ANN Accuracy: {ann_acc*100:.2f}%")

# =====================
# 5. Accuracy Comparison
# =====================
print("\n📊 ACCURACY COMPARISON")
print("="*40)
print(f"KNN            : {knn_acc*100:.2f}%")
print(f"Random Forest  : {rf_acc*100:.2f}%")
print(f"ANN            : {ann_acc*100:.2f}%")
without_ann = (knn_acc + rf_acc) / 2
print(f"\nWithout ANN avg: {without_ann*100:.2f}%")
print(f"With ANN       : {ann_acc*100:.2f}%")
print(f"Improvement    : +{(ann_acc - without_ann)*100:.2f}%")

# =====================
# 6. Confusion Matrices
# =====================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
class_names = le.classes_

for ax, (pred, title) in zip(axes, [
    (knn_pred, 'KNN'),
    (rf_pred, 'Random Forest'),
    (ann_pred, 'ANN')
]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, ax=ax, cmap='Blues', fmt='d',
                xticklabels=class_names,
                yticklabels=class_names)
    acc = accuracy_score(y_test, pred)
    ax.set_title(f'{title}\nAccuracy: {acc*100:.1f}%')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('models/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved!")

# =====================
# 7. Bar Chart
# =====================
plt.figure(figsize=(8, 5))
models_list = ['KNN', 'Random Forest', 'ANN']
accs = [knn_acc*100, rf_acc*100, ann_acc*100]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = plt.bar(models_list, accs, color=colors, width=0.5)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{acc:.2f}%', ha='center', fontweight='bold')
plt.ylim(0, 110)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.savefig('models/accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Accuracy chart saved!")

# =====================
# 8. Save Models
# =====================
pickle.dump(knn, open('models/knn_model.pkl', 'wb'))
pickle.dump(rf,  open('models/rf_model.pkl', 'wb'))
ann.save('models/ann_model.keras')
print("\n✅ All models saved!")