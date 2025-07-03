import torch
from tqdm import tqdm
import numpy as np

# Training function
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Train and validate model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function 
        optimizer: Optimizer
        device: Device to train on
        epochs: Number of epochs to train for
    """

    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                        leave=True, position=0, 
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for inputs, targets in train_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            train_acc = 100 * train_correct / train_total
            train_bar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{train_acc:.2f}%')
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total
        
        # # Validation phase
        # model.eval()
        # val_loss = 0.0
        # val_correct = 0
        # val_total = 0
        
        # val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]', 
        #               leave=True, position=0, 
        #               bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        # # Disable gradients during validation
        # with torch.no_grad():
        #     for inputs, targets in val_bar:
        #         inputs = inputs.to(device)
        #         targets = targets.to(device)
                
        #         outputs = model(inputs)
        #         loss = criterion(outputs, targets)
                
        #         val_loss += loss.item()
                
        #         # Calculate accuracy
        #         _, predicted = torch.max(outputs.data, 1)
        #         val_total += targets.size(0)
        #         val_correct += (predicted == targets).sum().item()
                
        #         val_acc = 100 * val_correct / val_total
        #         val_bar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{val_acc:.2f}%')
        
        # avg_val_loss = val_loss / len(val_loader)
        # avg_val_acc = 100 * val_correct / val_total
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Training Loss: {avg_train_loss:.4f}, Training Acc: {avg_train_acc:.2f}%')
        # print(f'  Validation Loss: {avg_val_loss:.4f}, Validation Acc: {avg_val_acc:.2f}%')
        
        # # Save model if validation accuracy improved
        # if avg_val_acc > best_val_acc:
        #     best_val_acc = avg_val_acc
        #     best_val_loss = avg_val_loss
        #     print(f'  Validation accuracy improved! Saving model...')
        torch.save(model.state_dict(), f'Models/traffic_signs/best_modelSEAME_epoch_{epoch+1}.pth')
        
        # # Also save every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     torch.save(model.state_dict(), f'Models/traffic_signs/checkpoint_epoch_{epoch+1}.pth')
    
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%, Best validation loss: {best_val_loss:.4f}')
    return model