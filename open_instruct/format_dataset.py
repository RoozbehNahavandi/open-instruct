from datasets import load_dataset
import random
import json


def format_daily_dialog_to_lima_format(output_path, max_samples=None, split='train'):
    """
    Convert the DailyDialog dataset to the specified LIMA-like format.
    Each sample alternates between combining dialog turns into the "user" role,
    separated by <EOU>, and using the next turn for the "assistant" role.

    Args:
        output_path (str): Path to save the formatted dataset.
        max_samples (int, optional): Limit the number of samples processed. Useful for debugging or testing.
        split (str): The dataset split to process ('train', 'validation', 'test').
    """
    # Load DailyDialog dataset
    dataset = load_dataset("daily_dialog")
    dialogs = dataset[split]["dialog"]
    formatted_data = []

    for idx, dialog in enumerate(dialogs):
        if max_samples and idx >= max_samples:
            break  # Stop if the maximum sample limit is reached
        print(f'Processing dialog {idx + 1}/{len(dialogs)}')

        if len(dialog) < 2:
            continue  # Skip dialogs with fewer than two turns

        for i in range(1, len(dialog)):
            # Combine turns up to i for the "user" role, separated by <EOU>
            user_dialog = " <EOU> ".join(dialog[:i]) + " <EOU>"
            assistant_response = dialog[i]  # The next turn for the "assistant" role

            # Format the entry
            formatted_data.append({
                "dataset": "daily_dialog",
                "id": f"daily_dialog_{idx}_turn_{i}",
                "messages": [
                    {"role": "user", "content": user_dialog},
                    {"role": "assistant", "content": assistant_response}
                ]
            })

    # Save the formatted data as a JSONL file
    with open(output_path, "w") as f:
        for entry in formatted_data:
            json.dump(entry, f)
            f.write("\n")

    print(f"Formatted dataset saved to {output_path}. Total samples: {len(formatted_data)}")




def format_daily_dialog_to_ultrafeedback(output_path, max_samples=None, split='train', context_size=5):
    """
    Convert the DailyDialog dataset to the UltraFeedback format, considering the context size.

    Args:
        output_path (str): Path to save the formatted dataset.
        max_samples (int, optional): Limit the number of samples processed. Useful for debugging or testing.
        split (str): Dataset split to process ('train', 'validation', or 'test').
        context_size (int): Number of past utterances to include in the context.
    """
    # Load DailyDialog dataset
    dataset = load_dataset("daily_dialog", split=split)
    dialogs = dataset["dialog"]
    formatted_data = []
    EOU_TOKEN = "<EOU>"

    # Precompute random responses for rejected (reduce repeated random sampling)
    all_responses = [utterance for dialog in dialogs for utterance in dialog]
    random_responses = random.choices(all_responses, k=len(all_responses))

    for idx, dialog in enumerate(dialogs):
        if max_samples and idx >= max_samples:
            break  # Stop if the maximum sample limit is reached
        print(f'Processing dialog {idx + 1}/{len(dialogs)}')

        if len(dialog) < 2:
            continue  # Skip dialogs with fewer than two turns

        contexts = []  # Keep track of the context
        for i, utterance in enumerate(dialog):
            if len(contexts) >= context_size:
                # Construct the "prompt" using the last `context_size` turns
                context = f" {EOU_TOKEN} ".join(contexts[-context_size:]) + f" {EOU_TOKEN}"
                # Add the target response
                target = utterance + EOU_TOKEN

                # Create the `chosen` response
                chosen_response = [
                    {"content": context, "role": "user"},
                    {"content": target, "role": "assistant"}
                ]

                # Create the `rejected` response
                random_rejected = random.choice(random_responses)
                rejected_response = [
                    {"content": context, "role": "user"},
                    {"content": random_rejected, "role": "assistant"}
                ]

                # Add to the formatted dataset
                formatted_data.append({
                    "prompt": context,
                    "chosen": chosen_response,
                    "rejected": rejected_response
                })

            # Update the contexts for the next iteration
            contexts.append(utterance)

    # Save the formatted data as a JSON file
    with open(output_path, "w") as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Formatted dataset saved to {output_path}. Total samples: {len(formatted_data)}")


def format_daily_dialog_to_ultrafeedback_with_metadata(output_path, max_samples=None, split='train', context_size=3):
    """
    Convert the DailyDialog dataset to the UltraFeedback format, including 'emotion' and 'intent' metadata under 'meta_data'.

    Args:
        output_path (str): Path to save the formatted dataset.
        max_samples (int, optional): Limit the number of samples processed. Useful for debugging or testing.
        split (str): Dataset split to process ('train', 'validation', or 'test').
        context_size (int): Number of past utterances to include in the context.
    """
    # Load DailyDialog dataset
    dataset = load_dataset("daily_dialog", split=split)
    dialogs = dataset["dialog"]
    emotions = dataset["emotion"]
    intents = dataset["act"]
    formatted_data = []
    EOU_TOKEN = "<EOU>"

    # Precompute random responses for rejected (reduce repeated random sampling)
    all_responses = [utterance for dialog in dialogs for utterance in dialog]
    random_responses = random.choices(all_responses, k=len(all_responses))

    for idx, dialog in enumerate(dialogs):
        if max_samples and idx >= max_samples:
            break  # Stop if the maximum sample limit is reached
        print(f'Processing dialog {idx + 1}/{len(dialogs)}')

        if len(dialog) < 2:
            continue  # Skip dialogs with fewer than two turns

        contexts = []  # Keep track of the context
        for i, utterance in enumerate(dialog):
            if len(contexts) >= context_size:
                # Construct the "prompt" using the last `context_size` turns
                context = f" {EOU_TOKEN} ".join(contexts[-context_size:]) + f" {EOU_TOKEN}"
                # Add the target response
                target = utterance + EOU_TOKEN

                # Create the `chosen` response
                chosen_response = [
                    {"content": context, "role": "user"},
                    {"content": target, "role": "assistant"}
                ]

                # Create the `rejected` response
                random_rejected = random.choice(random_responses)
                rejected_response = [
                    {"content": context, "role": "user"},
                    {"content": random_rejected, "role": "assistant"}
                ]

                # Add to the formatted dataset
                formatted_data.append({
                    "prompt": context,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "meta_data": {
                        "emotion": [emotions[idx][i]],  # Add corresponding emotion
                        "intent": [intents[idx][i]]    # Add corresponding intent
                    }
                })

            # Update the contexts for the next iteration
            contexts.append(utterance)

    # Save the formatted data as a JSONL file
    with open(output_path, "w") as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Formatted dataset saved to {output_path}. Total samples: {len(formatted_data)}")


if __name__ == "__main__":
    # Specify the output file path and limit samples for debugging
    split = 'test'
    output_file = f"daily_dialog_ultrafeedback_metadata_{split}.jsonl"
    max_samples = 20_000  # Set to a small number (e.g., 100) for testing/debugging
    context_size = 5  # Number of past utterances to include in the context

    # Run the conversion
    format_daily_dialog_to_ultrafeedback_with_metadata(output_file, max_samples=max_samples, split=split, context_size=context_size)