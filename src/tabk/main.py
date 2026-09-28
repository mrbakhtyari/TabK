from sklearn.datasets import load_iris

from tabk.architecture import load_inference_context, predict_single
from tabk.utils import apply_standard_scaling


def main():
    # Load the Iris dataset
    X, _ = load_iris(return_X_y=True)
    X_scaled = apply_standard_scaling(X)

    # Download the pretrained 5-fold ensemble from the Hugging Face Hub
    ctx = load_inference_context()

    # Predict k in a single forward pass
    predicted_k = predict_single(ctx, X_scaled)
    print(f"Predicted number of clusters: {predicted_k}")


if __name__ == "__main__":
    main()
