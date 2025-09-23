from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

model.eval()
train_preds, train_probs = [], []
with torch.no_grad():
    for xb, yb in DataLoader(train_dataset, batch_size=256, shuffle=False):
        xb = xb.to(device)
        out = model(xb)
        prob = torch.sigmoid(out).cpu().numpy()
        pred = (prob >= 0.5).astype(int).ravel()
        train_probs.extend(prob.ravel().tolist())
        train_preds.extend(pred.tolist())

print("\n=== Train Set Report ===")
print(classification_report(y_train, train_preds, digits=4))
print("ROC AUC:", roc_auc_score(y_train, train_probs))
print("Confusion matrix:\n", confusion_matrix(y_train, train_preds))

print("\n=== Test Set Report ===")
print(classification_report(y_test, preds, digits=4))
print("ROC AUC:", roc_auc_score(y_test, probs))
print("Confusion matrix:\n", confusion_matrix(y_test, preds))
