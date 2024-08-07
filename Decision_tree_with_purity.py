import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text,_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv('drug200_1722406166375.csv')

X = data[['Age', 'Sex', 'BP', 'Cholesterol']]
y = data['Drug']

X = pd.get_dummies(X, columns=['Sex', 'BP', 'Cholesterol'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, criterion, min_purity):
        super().__init__(criterion=criterion)
        self.min_purity = min_purity

    def fit(self, X, y):
        super().fit(X, y)
        self._prune_tree()

    def _prune_tree(self):
        if not hasattr(self, 'tree_'):
            return

        def prune_node(node):
            if self.tree_.feature[node] == _tree.TREE_LEAF:
                return

            left_child = self.tree_.children_left[node]
            right_child = self.tree_.children_right[node]

            if left_child == right_child: 
                return

            # Recursively prune children
            prune_node(left_child)
            prune_node(right_child)

            # Check if the node should be pruned
            if self._should_prune(node, left_child, right_child):
                self.tree_.feature[node] = _tree.TREE_LEAF
                self.tree_.threshold[node] = -2
                self.tree_.value[node] = self.tree_.value[left_child] + self.tree_.value[right_child]
                self.tree_.children_left[node] = self.tree_.children_right[node] = _tree.TREE_LEAF

        prune_node(0)  # Start pruning from the root node

    def _should_prune(self, node, left_child, right_child):
        left_purity = self._calculate_purity(left_child)
        right_purity = self._calculate_purity(right_child)
        return left_purity >= self.min_purity and right_purity >= self.min_purity

    def _calculate_purity(self, node):
        # Calculate the purity of the given node
        if self.tree_.feature[node] == _tree.TREE_LEAF:
            total_samples = np.sum(self.tree_.value[node])
            max_class_samples = np.max(self.tree_.value[node])
            return max_class_samples / total_samples if total_samples > 0 else 0
        return 0

clf = CustomDecisionTreeClassifier(criterion='entropy', min_purity=0.8)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(export_text(clf, feature_names=list(X.columns)))

