import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification

# Custom Dataset Class for Training
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Training Function
def train_model():
    # Example Data for Training
    texts = ["I love programming.", "I hate bugs.", "Debugging is amazing!", "This is horrible."]
    labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

    # Load Tokenizer and Model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Prepare Dataset and Dataloader
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Optimizer and Device Setup
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Loop
    model.train()
    for epoch in range(15):  # Adjust the number of epochs as needed
        total_loss = 0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

    # Save Fine-Tuned Model and Tokenizer
    tokenizer.save_pretrained('./fine_tuned_bert')
    model.save_pretrained('./fine_tuned_bert')
    print("Fine-tuned JokerBERT saved to './fine_tuned_bert'")

# Main Execution
if __name__ == "__main__":
    train_model()
