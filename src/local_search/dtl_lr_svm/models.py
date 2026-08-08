import numpy as np

def f1_macro(y_true, y_pred):
    classes = np.unique(y_true)
    f1_scores = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores)

class StableLogisticRegression:
    def __init__(self, learning_rate=0.8, iterations=25000, lambda_reg=0.00005):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.theta = None

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        
        w0 = 1.0
        w1 = 2.2
        weights = np.where(y == 1, w1, w0)

        for _ in range(self.iterations):
            h = self.sigmoid(np.dot(X, self.theta))
            error = weights * (h - y)
            gradient = (1/m) * np.dot(X.T, error)
            
            reg_term = (self.lambda_reg / m) * self.theta
            reg_term[0] = 0
            gradient += reg_term

            self.theta -= self.learning_rate * gradient

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.theta))

class StableLogisticRegressionVis(StableLogisticRegression):
    def __init__(self, learning_rate=0.8, iterations=25000, lambda_reg=0.00005):
        super().__init__(learning_rate, iterations, lambda_reg)
        self.loss_history = []
        self.theta_history = []
    
    def compute_cost(self, X, y, weights):
        m = len(y)
        h = self.sigmoid(np.dot(X, self.theta))
        epsilon = 1e-15
        cost = -(1/m) * np.sum(weights * (y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)))
        reg_cost = (self.lambda_reg / (2 * m)) * np.sum(self.theta[1:] ** 2)
        return cost + reg_cost

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        w0, w1 = 1.0, 2.2
        weights = np.where(y == 1, w1, w0)

        for i in range(self.iterations):
            h = self.sigmoid(np.dot(X, self.theta))
            error = weights * (h - y)
            gradient = (1/m) * np.dot(X.T, error)
            
            reg_term = (self.lambda_reg / m) * self.theta
            reg_term[0] = 0
            gradient += reg_term

            self.theta -= self.learning_rate * gradient
            
            if i % 500 == 0:
                self.loss_history.append(self.compute_cost(X, y, weights))
                self.theta_history.append([self.theta[1], self.theta[2]])

class NodeDT:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeScratch:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if num_samples <= 1 or depth >= self.max_depth or len(np.unique(y)) == 1:
            leaf_value = np.bincount(y).argmax()
            return NodeDT(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y, num_features)
        
        if best_feat is None:
            leaf_value = np.bincount(y).argmax()
            return NodeDT(value=leaf_value)

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return NodeDT(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, num_features):
        best_gini = 1.0
        split_idx, split_thresh = None, None
        
        for feat_idx in range(num_features):
            thresholds = np.unique(X[:, feat_idx])
            if len(thresholds) > 10:
                thresholds = np.percentile(thresholds, [25, 50, 75])
                
            for thresh in thresholds:
                left_y = y[X[:, feat_idx] <= thresh]
                right_y = y[X[:, feat_idx] > thresh]
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                gini = self._gini_impurity(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_thresh = thresh
                    
        return split_idx, split_thresh

    def _gini_impurity(self, left_y, right_y):
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = len(right_y) / (len(left_y) + len(right_y))
        
        def gini(y):
            if len(y) == 0: return 0
            p1 = np.sum(y == 1) / len(y)
            p0 = 1 - p1
            return 1 - (p0**2 + p1**2)
            
        return p_left * gini(left_y) + p_right * gini(right_y)