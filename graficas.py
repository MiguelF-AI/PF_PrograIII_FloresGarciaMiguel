import matplotlib.pyplot as plt

def GraficarResultados(historial):
    # Extraer datos del historial de entrenamiento
    acc = historial.history['accuracy']
    val_acc = historial.history['val_accuracy']
    loss = historial.history['loss']
    val_loss = historial.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Crear una figura con dos subgráficas 
    plt.figure(figsize=(14, 5))

    # Gráfica de Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Precisión Entrenamiento')
    plt.plot(epochs, val_acc, 'r-', label='Precisión Validación')
    plt.title('Accuracy por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    # Gráfica de Pérdida 
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Pérdida Entrenamiento')
    plt.plot(epochs, val_loss, 'r-', label='Pérdida Validación')
    plt.title('Pérdida por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)

    # Mostrar Gráficas
    plt.tight_layout()
    plt.show()