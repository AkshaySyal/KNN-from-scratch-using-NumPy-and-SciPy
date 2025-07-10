# KNN-from-scratch-using-NumPy-and-SciPy
This project involved preprocessing the MNIST dataset (combining, flattening, and binarizing images, then converting to sparse format) and implementing a K-Nearest Neighbors (KNN) classifier from scratch using cosine similarity to classify images, reporting performance on training, validation, and test sets

# Problem Statement
![image](https://github.com/user-attachments/assets/aeff55b7-bf3f-460c-bec4-163719a13d03)

# Solution Summary

- Preprocessed the **MNIST dataset** and implemented a **K-Nearest Neighbors (KNN) classifier** from scratch.

- **Data Preprocessing**:
  - Combined training and testing images into a **70,000-image dataset**.
  - Flattened images from **28x28 to 784-pixel vectors**.
  - Binarized pixel values: set all nonzero pixels to 1.
  - Converted data to **sparse csr_matrix format** for memory efficiency.

- **KNN Classifier Implementation**:
  - Built KNN classifier **without using scikit-learn** (external support libraries allowed for data structures).
  - Used **cosine similarity** as the distance metric.
  - Dataset split:
    - Training: indices 0–55,999
    - Validation: indices 56,000–62,999
    - Testing: indices 63,000–69,999

- **Key Insights**:
  - KNN “training” involves **storing data and computing nearest neighbors**, with no iterative or gradient-based updates.
  - Performance evaluated by predicting labels on subsets.

- **Results**:
  - Training accuracy: **85.0%**
  - Validation accuracy: **59.0%**
  - Testing accuracy: **65.0%**

- **Conclusion**:
  - KNN offers simplicity but shows limited generalization compared to more sophisticated models, especially on unseen test data.

- Used **NumPy** and **SciPy** for data preprocessing and implementing the **K-Nearest Neighbors (KNN) classifier**.

- **NumPy Usage**:
  - Combined `train_images` and `test_images` using `np.concatenate` → created `all_imgs` of shape **(70000, 28, 28)**.
  - Combined `train_labels` and `test_labels` into `all_labels`.
  - Flattened images from **28x28** to **784-pixel vectors** with `all_imgs.reshape`, resulting in `all_imgs_flat` of shape **(70000, 784)**.
  - Binarized pixel values: set all pixels > 0 to 1 using array indexing (`all_imgs_flat[all_imgs_flat > 0] = 1`).
  - In `cosine_similarity`:
    - Calculated vector magnitudes with `np.sqrt(mag_arr[idx])`.
    - Summed pixel values with `np.sum(all_imgs_flat, axis=1)`.

- **SciPy Usage**:
  - Used `scipy.sparse.csr_matrix` to convert `all_imgs_flat` into a **sparse matrix** (`sparse_imgs_flat`) for memory efficiency.
  - Sparse matrix shape: **(70000, 784)**.
  - Computed dot products for cosine similarity efficiently on sparse data (`sparse_imgs_flat.dot(sparse_imgs_flat.T)`).

- **Key Insight**:
  - Sparse matrix representation was crucial due to the high proportion of zeros in the binarized dataset, significantly improving computational efficiency.
