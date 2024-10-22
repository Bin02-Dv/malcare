from django.shortcuts import render
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
from django.http import JsonResponse

# Create your views here.

def index(request):
    if request.method == 'POST' and 'img' in request.FILES:
        img_file = request.FILES['img']  # Fetch the uploaded image
        
        # Load the pre-trained model
        model = load_model('C:\\Users\\ALAMEEN\\Documents\\Documents\\my_projects\\malcareProject\\malcareProject\\mal_model.keras')
        
        # Convert the uploaded image (InMemoryUploadedFile) to a file-like object (BytesIO)
        img_bytes = img_file.read()  # Read the image file as bytes
        img_con = Image.open(io.BytesIO(img_bytes))  # Open the image using PIL
        img_con = img_con.resize((128, 128))  # Resize image
        
        # Convert the uploaded image to a format suitable for the model
        # img = image.load_img(img_con, target_size=(128, 128))  # Load image with specified size
        img_array = image.img_to_array(img_con)  # Convert image to array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image
        
        # Make prediction using the model
        prediction = model.predict(img_array)
        
        # Interpret the prediction result
        # result = "Uninfected" if prediction[0] > 0.5 else "Infected"
        
        if prediction[0] > 0.5:
            model_response = 'This sample is Uninfected...'
            return JsonResponse({'success': True, 'message': model_response})
        else:
            model_response = 'This sample is Infected...'
            return JsonResponse({'success': True, 'message': model_response})
        
    # If not POST request or no file is uploaded, return the default page
    return render(request, "index.html")
