true_labels = [
    'refund',
    'refund',
    'payment',
    'shipping',
]

predicted_labels = [
    'refund',
    'refund',
    'refund',
    'refund',
]

correct_count = 0

for true_label, predicted_label in zip(
    true_labels,
    predicted_labels,
):
    is_correct = true_label == predicted_label

    if is_correct:
        correct_count += 1

    print(
        f'True: {true_label}, '
        f'Predicted: {predicted_label}, '
        f'Correct: {is_correct}'
    )

accuracy = correct_count / len(true_labels)

print('\n--- Accuracy ---')
print('Correct count:', correct_count)
print('Total count:', len(true_labels))
print('Accuracy:', accuracy)


target_label = 'refund'

predicted_target_count = 0
correct_target_count = 0

for true_label, predicted_label in zip(
    true_labels,
    predicted_labels,
):
    if predicted_label == target_label:
        predicted_target_count += 1

        if true_label == target_label:
            correct_target_count += 1

precision = (
    correct_target_count / predicted_target_count
    if predicted_target_count > 0
    else 0.0
)

print('\n--- Refund Precision ---')
print('Predicted as refund:', predicted_target_count)
print('Correctly predicted as refund:', correct_target_count)
print('Precision:', precision)

target_label = 'refund'

actual_target_count = 0
correct_target_count = 0

for true_label, predicted_label in zip(
    true_labels,
    predicted_labels,
):
    if true_label == target_label:
        actual_target_count += 1

        if predicted_label == target_label:
            correct_target_count += 1

recall = (
    correct_target_count / actual_target_count
    if actual_target_count > 0
    else 0.0
)

print('\n--- Refund Recall ---')
print('Actual refund count:', actual_target_count)
print('Correctly predicted as refund:', correct_target_count)
print('Recall:', recall)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0.0
)

print('\n--- Refund Metrics ---')
print('Precision:', precision)
print('Recall:', recall)
print('F1:', round(f1, 4))

