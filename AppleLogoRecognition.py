import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects



def myConnectedComponentsWithStats(binary_image, connectivity=8):
    """
    Custom implementation of cv2.connectedComponentsWithStats using scipy.ndimage.
    Returns:
    - num_labels: Total number of labels (including background 0).
    - labels: A 2D array of the same size as input_image, where each pixel's value
              is the label of the connected component to which it belongs.
    - stats: A NumPy array with statistics for each label,
             similar to OpenCV's output format (x, y, width, height, area).
             Note: cv2.connectedComponentsWithStats provides additional stats like CC_STAT_LEFT,
             CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA.
             This implementation provides them in that order.
    
    """
    # Label connected components
    labels, num_features = label(binary_image, structure=np.ones((3,3)) if connectivity == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int))

    # Initialize stats array
    stats = np.zeros((num_features + 1, 5), dtype=np.int32) # x, y, width, height, area
  
    # Get overall image dimensions for calculating moments for each component
    img_height, img_width = binary_image.shape
    overall_y_indices, overall_x_indices = np.indices(binary_image.shape)

    # Iterate through each labeled component (1 to num_features)
    for i in range(1, num_features + 1):
        component_mask = (labels == i) # This is a boolean mask for the current component

        # Calculate bounding box
        rows, cols = np.where(component_mask) # Get pixel coordinates for the current component
        
        if len(rows) > 0: # Check if the component has any pixels
            y_min, y_max = np.min(rows), np.max(rows)
            x_min, x_max = np.min(cols), np.max(cols)
            
            x_bbox = x_min
            y_bbox = y_min
            w_bbox = x_max - x_min + 1
            h_bbox = y_max - y_min + 1
            area = np.sum(component_mask) # Area is just the sum of pixels in the mask

            stats[i, :] = [x_bbox, y_bbox, w_bbox, h_bbox, area]

            # Calculate centroid using calcNormalMoment on the isolated component
            # We need to create a temporary image that is just this component's mask
            # and then call calcNormalMoment on it.
            # Crop the component mask to its bounding box to pass to calcNormalMoment efficiently
            cropped_component_mask = binary_image[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox] * component_mask[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox]
            
            # Recalculate relative coordinates for the cropped mask for calcNormalMoment
            # This is slightly inefficient as calcNormalMoment does its own np.indices
            # A more direct way would be to pass the actual global x,y and component_mask
            # to a custom moment function that takes mask and global coordinates.
            # For simplicity, and since calcNormalMoment is already structured,
            # we'll use it this way, understanding the input image should be the cropped mask.

            # We need to ensure that calcNormalMoment is working on the specific component's pixels.
            # The original calcNormalMoment binarizes its input (image > 0).astype(np.uint8).
            # So, we pass the `component_mask` (which is boolean or 0/1 uint8) to it directly.
            # However, calcNormalMoment also calculates coordinates starting from (0,0) of the *passed image*.
            # For centroid, we need coordinates relative to the *overall image*.

            # Let's adjust calcNormalMoment to take a global offset if needed, or
            # more simply, calculate moments based on the global coordinate grids
            # multiplied by the component_mask itself.

        else:
            stats[i, :] = [0, 0, 0, 0, 0]
           

    return num_features + 1, labels, stats # +1 for background label 0

def my_rectangle(img, pt1, pt2, color, thickness=1):
    """
    Custom implementation of cv2.rectangle to draw a rectangle on an image.
    img: NumPy array representing the image.
    pt1: Tuple (x1, y1) - top-left corner.
    pt2: Tuple (x2, y2) - bottom-right corner.
    color: Tuple (B, G, R) for color image, or scalar for grayscale.
    thickness: Integer, thickness of the rectangle lines.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Ensure coordinates are within image bounds
    y_min, y_max = max(0, min(y1, y2)), min(img.shape[0], max(y1, y2) + 1)
    x_min, x_max = max(0, min(x1, x2)), min(img.shape[1], max(x1, x2) + 1)

    # Draw horizontal lines
    for t in range(thickness):
        if y_min + t < img.shape[0]:
            img[y_min + t, x_min:x_max] = color
        if y_max - 1 - t >= 0:
            img[y_max - 1 - t, x_min:x_max] = color
    
    # Draw vertical lines
    for t in range(thickness):
        if x_min + t < img.shape[1]:
            img[y_min:y_max, x_min + t] = color
        if x_max - 1 - t >= 0:
            img[y_min:y_max, x_max - 1 - t] = color
    
    return img

def to_log_hu(val):
    return -1 * np.copysign(1.0, val) * np.log10(abs(val)) if val != 0 else 0

def calcNormalMoment(image,number1,number2):
    image = image.astype(np.float64) 
    y, x = np.indices(image.shape)
    return np.sum((x ** number1) * (y ** number2) * image)
    

def calcCentralMoment(image,number1,number2):
    #"M(x,y)"
    ii=calcNormalMoment(image,1,0) / calcNormalMoment(image,0,0)
    jj=calcNormalMoment(image,0,1) / calcNormalMoment(image,0,0)
  
    if(number1==0 and number2==0):
        centralMoment=calcNormalMoment(image,0,0)
    elif(number1==2 and number2==0):
        centralMoment=calcNormalMoment(image,2,0) - np.pow(calcNormalMoment(image,1,0),2) / calcNormalMoment(image,0,0)
    elif(number1==0 and number2==2):
        centralMoment=calcNormalMoment(image,0,2) - np.pow(calcNormalMoment(image,0,1),2) / calcNormalMoment(image,0,0)
    elif(number1==1 and number2==1):
        centralMoment=calcNormalMoment(image,1,1) - calcNormalMoment(image,1,0) * calcNormalMoment(image,0,1) / calcNormalMoment(image,0,0)
    elif(number1==2 and number2==1):
        centralMoment=calcNormalMoment(image,2,1) - 2 * calcNormalMoment(image,1,1) * ii-calcNormalMoment(image,2,0) * jj + 2* calcNormalMoment(image,0,1) * np.pow(ii,2)
    elif(number1==1 and number2==2):
        centralMoment=calcNormalMoment(image,1,2) - 2 * calcNormalMoment(image,1,1) * jj-calcNormalMoment(image,0,2) * ii + 2* calcNormalMoment(image,1,0) * np.pow(jj,2)
    elif(number1==3 and number2==0):
        centralMoment=calcNormalMoment(image,3,0) - 3 * calcNormalMoment(image,2,0) * ii + 2 * calcNormalMoment(image,1,0) * np.pow(ii,2)
    elif(number1==0 and number2==3):
        centralMoment=calcNormalMoment(image,0,3) - 3 * calcNormalMoment(image,0,2) * jj + 2 * calcNormalMoment(image,0,1) * np.pow(jj,2)
    else:
        print("this central moment is not available yet")

    return centralMoment

def calcNormalizedCentralMoment(image,number1,number2):
    #"N(x,y)"
    normalizedCentralMoment = calcCentralMoment(image,number1,number2) / np.pow(calcCentralMoment(image,0,0),1 + (number1 + number2) / 2)

    return normalizedCentralMoment

def calcHuMoment(image,number):
    #"M(x)"
    if(number==0):
        huMoment = calcNormalizedCentralMoment(image,2,0) + calcNormalizedCentralMoment(image,0,2)
    elif(number==1):
        huMoment = np.pow((calcNormalizedCentralMoment(image,2,0) - calcNormalizedCentralMoment(image,0,2)),2) + 4 * np.pow(calcNormalizedCentralMoment(image,1,1),2)
    elif(number==2):
        huMoment = np.pow(calcNormalizedCentralMoment(image,3,0) - 3 * calcNormalizedCentralMoment(image,1,2),2) + np.pow(3*calcNormalizedCentralMoment(image,2,1)-calcNormalizedCentralMoment(image,0,3),2)
    elif(number==3):
        huMoment = np.pow(calcNormalizedCentralMoment(image,3,0) + calcNormalizedCentralMoment(image,1,2),2) + np.pow(calcNormalizedCentralMoment(image,2,1)+calcNormalizedCentralMoment(image,0,3),2)
    elif(number==4):
        huMoment = (calcNormalizedCentralMoment(image,3,0) - 3 * calcNormalizedCentralMoment(image,1,2)) * (calcNormalizedCentralMoment(image,3,0) + calcNormalizedCentralMoment(image,1,2)) * (np.pow(calcNormalizedCentralMoment(image,3,0) + calcNormalizedCentralMoment(image,1,2),2) - 3 * np.pow(calcNormalizedCentralMoment(image,2,1) + calcNormalizedCentralMoment(image,0,3),2)) + (3 * calcNormalizedCentralMoment(image,2,1) - calcNormalizedCentralMoment(image,0,3)) * (calcNormalizedCentralMoment(image,2,1) + calcNormalizedCentralMoment(image,0,3)) * (3 * np.pow(calcNormalizedCentralMoment(image,3,0) + calcNormalizedCentralMoment(image,1,2),2) - np.pow(calcNormalizedCentralMoment(image,2,1) + calcNormalizedCentralMoment(image,0,3),2))
    elif(number==5):
        huMoment = (calcNormalizedCentralMoment(image,2,0) - calcNormalizedCentralMoment(image,0,2)) * (np.pow(calcNormalizedCentralMoment(image,3,0) + calcNormalizedCentralMoment(image,1,2),2) -  np.pow(calcNormalizedCentralMoment(image,2,1) + calcNormalizedCentralMoment(image,0,3),2)) + 4 * calcNormalizedCentralMoment(image,1,1) * (calcNormalizedCentralMoment(image,3,0) + calcNormalizedCentralMoment(image,1,2)) * (calcNormalizedCentralMoment(image,2,1) + calcNormalizedCentralMoment(image,0,3))
    elif(number==6):
        huMoment = (3 * calcNormalizedCentralMoment(image,2,1) - calcNormalizedCentralMoment(image,0,3)) * (calcNormalizedCentralMoment(image,3,0) + calcNormalizedCentralMoment(image,1,2)) * (np.pow((calcNormalizedCentralMoment(image,3,0) + calcNormalizedCentralMoment(image,1,2)),2) - 3 * np.pow((calcNormalizedCentralMoment(image,2,1) + calcNormalizedCentralMoment(image,0,3)),2)) - (calcNormalizedCentralMoment(image,3,0) -3 * calcNormalizedCentralMoment(image,1,2)) * (calcNormalizedCentralMoment(image,2,1) + calcNormalizedCentralMoment(image,0,3)) * (3 * np.pow(calcNormalizedCentralMoment(image,3,0) + calcNormalizedCentralMoment(image,1,2),2) - np.pow(calcNormalizedCentralMoment(image,2,1) + calcNormalizedCentralMoment(image,0,3),2))
    else:
        print("this hu moment is not available yet")
    return huMoment

def mask_in_range(hsv_img, lower, upper):
    """
    Własna wersja cv2.inRange – zwraca maskę binarną (0 lub 255) tam,
    gdzie każdy piksel HSV mieści się w podanym zakresie.
    
    hsv_img: obraz HSV w kształcie (wys, szer, 3)
    lower, upper: krotki numpy.array z 3 wartościami (np. [20, 100, 100])
    """
    # Tworzymy maskę logiczną: True tam, gdzie piksel spełnia wszystkie warunki
    condition = (
        (hsv_img[:, :, 0] >= lower[0]) & (hsv_img[:, :, 0] <= upper[0]) &
        (hsv_img[:, :, 1] >= lower[1]) & (hsv_img[:, :, 1] <= upper[1]) &
        (hsv_img[:, :, 2] >= lower[2]) & (hsv_img[:, :, 2] <= upper[2])
    )

    # Zamieniamy True/False na obraz 0/255 (uint8)
    mask = np.where(condition, 255, 0).astype(np.uint8)
    return mask

def detect_rainbow_colors(hsv_img,threshold=50):
    # funkcja do detekcji kolorów tęczy w formacie barw HSV
    positions={}

    color_ranges = {
    'green': [(np.array([40, 100, 100]), np.array([85, 255, 255]))],
    'yellow': [(np.array([25, 100, 100]), np.array([35, 255, 255]))],
    'orange': [(np.array([10, 100, 100]), np.array([25, 255, 255]))],
    'red': [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
    ],
    'purple': [(np.array([130, 50, 70]), np.array([160, 255, 255]))],
    'blue': [(np.array([90, 100, 100]), np.array([130, 255, 255]))],
    }

    for color_name,ranges in color_ranges.items():
        mask=0
        for lower,upper in ranges:
            mask=mask_in_range(hsv_img,lower,upper)
        # xs chyba nie jest potrzebne bo tu badam kolejnosc paskow w kolejnosci pionowej
        # moze w jakiejs zaawansowanej implementacji dodam tez patrzenie po skladowej poziomej
        ys, xs = np.where(mask == 255)
        
        if len(ys) > threshold: # co najmniej 50 pikseli koloru
           
            avg_y = np.median(ys)

            #print("Srednia wartosc dla ",color_name," Wynosi ",avg_y," calosc ---> ",ys)
            positions[color_name] = avg_y
        
    return positions

def is_rainbow_in_order(detected_colors):
    # funkcja do sprawdzenia czy kolory są po kolei
    isOrder=False
    correct_order = ['green', 'yellow', 'orange', 'red', 'purple', 'blue']
 
    #if(len(detected_colors)!=len(correct_order)):
        #nie wykryto wszystkich barw teczy
        #print(len(detected_colors))
        #print(len(correct_order))
     #   return False
    
    #for name, value in detected_colors.items():
     #   print(name)
      #  print(value)
    
    # sortowanie wykrytych kolorow tylko po wsp Y (na samej gorze po lewo jest 0,0 wiec powinno byc rosnąco)
    # tzn ze green ma najmniejsza wartosc a niebieski najwieksza
    sorted_colors = sorted(detected_colors.items(), key=lambda x: x[1])
    '''print("AAAAAAAAAA")
    for name, value in sorted_colors:
        print(name)
        print(value)
    '''
    matchedOrder=0
    for i in range (0,len(sorted_colors)):
        sorted_name,sorted_value = sorted_colors[i]
        if(correct_order[i]==sorted_name):
            matchedOrder=matchedOrder + 1
        #print(correct_order[i])
        #print("ffffff")
        #print(sorted_colors[i])
    if(matchedOrder>=5):
        isOrder=True
    # 5 a nie wszystkie 7 kolorow, wiekszy bufor, moge nie patrzec tak stricte na wszystkie kolory
    
    return isOrder

#print( cv.__version__ )
# tu "roboczo" wczytuje sobie zdj i tworze maski do kazdego koloru zeby "zobaczyc" te maski
img = cv.imread("Apple/AppleLogo4.jpg",1)
hsv_img=cv.cvtColor(img, cv.COLOR_BGR2HSV)
#cv.imshow("img",hsv_img)
cv.imshow("img",img)
#print(detect_rainbow_colors(hsv_img,100))

lower_rgb = np.array([200, 200, 0])
upper_rgb = np.array([255, 255, 100])
mask_rgb = cv.inRange(img, lower_rgb, upper_rgb)


green_lower = np.array([40, 100, 100]) # Hue=40°, S=100, V=100
green_upper = np.array([85, 255, 255])

yellow_lower = np.array([20, 100, 100]) 
yellow_upper = np.array([35, 255, 255])  

orange_lower = np.array([10, 100, 100])
orange_upper = np.array([25, 255, 255])

red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([179, 255, 255])

purple_lower = np.array([130, 50, 70])
purple_upper = np.array([160, 255, 255])

blue_lower = np.array([90, 100, 100])
blue_upper = np.array([130, 255, 255])


mask_hsv_green=mask_in_range(hsv_img, green_lower, green_upper)
mask_hsv_yellow = mask_in_range(hsv_img, yellow_lower, yellow_upper)
mask_hsv_orange=mask_in_range(hsv_img, orange_lower, orange_upper)
mask_hsv_red=mask_in_range(hsv_img, red_lower, red_upper)
mask_hsv_red2=mask_in_range(hsv_img, red_lower2, red_upper2)
mask_hsv_red = cv.bitwise_or(mask_hsv_red, mask_hsv_red2)
mask_hsv_purple=mask_in_range(hsv_img, purple_lower, purple_upper)
mask_hsv_blue=mask_in_range(hsv_img, blue_lower, blue_upper)

#pokazywanie tych masek
plt.figure(figsize=(12, 5))

plt.subplot(1, 7, 1)
plt.title("Maska hsv green")
plt.imshow(mask_hsv_green, cmap='gray')
plt.axis("off")

plt.subplot(1, 7, 2)
plt.title("Maska hsv yellow")
plt.imshow(mask_hsv_yellow, cmap='gray')
plt.axis("off")

plt.subplot(1, 7, 3)
plt.title("Maska hsv orange")
plt.imshow(mask_hsv_orange, cmap='gray')
plt.axis("off")

plt.subplot(1, 7, 4)
plt.title("Maska hsv red")
plt.imshow(mask_hsv_red, cmap='gray')
plt.axis("off")

plt.subplot(1, 7,5)
plt.title("Maska hsv purple")
plt.imshow(mask_hsv_purple, cmap='gray')
plt.axis("off")

plt.subplot(1, 7, 6)
plt.title("Maska hsv blue")
plt.imshow(mask_hsv_blue, cmap='gray')
plt.axis("off")

#polaczenie masek w jedno
combined_mask = (
    (mask_hsv_green > 0) |
    (mask_hsv_yellow > 0) |
    (mask_hsv_orange > 0) |
    (mask_hsv_red > 0) |
    (mask_hsv_purple > 0) |
    (mask_hsv_blue > 0)
).astype(np.uint8) * 255

plt.subplot(1, 7, 7)
plt.title("maska all")
plt.imshow(combined_mask, cmap='gray')


print(is_rainbow_in_order(detect_rainbow_colors(hsv_img,100)))


print("\n--- Analiza komponentów spójnych dla logo Apple ---")

#this needs to be changed so that its no from cv
num_labels, labels, stats  = myConnectedComponentsWithStats(combined_mask, 8)

# Znajdź największy komponent (pomijając tło, które ma label=0)
max_area = 0
apple_logo_label = -1
# Jeśli obraz jest pusty lub tylko tło, num_labels może być 1
if num_labels > 1:
    for i in range(1, num_labels): # Iteracja od 1, aby pominąć tło (label 0)
        current_area = stats[i, 4]
        if current_area > max_area:
            max_area = current_area
            apple_logo_label = i

if apple_logo_label != -1:
    print(f"Wykryto potencjalne logo Apple (label: {apple_logo_label}, powierzchnia: {max_area} pikseli)")

    # maska zawierajaca tylko apple
    apple_mask = (labels == apple_logo_label).astype(np.uint8) * 255

    # Pobierz współrzędne ramki ograniczającej logo
    # tu musze wywalic cv i dac wlasna funkcje
    x = stats[apple_logo_label, 0] # Left
    y = stats[apple_logo_label, 1] # Top
    w = stats[apple_logo_label, 2] # Width
    h = stats[apple_logo_label, 3] # Height


  


    #
    cropped_logo_img = hsv_img[y : y + h, x : x + w]
    #cv.imshow("Wyciete Logo Apple (Oryginalny Obraz)", cropped_logo_img)
    #print("teeeeeeeeeeeeeeeeeest")
    #print(is_rainbow_in_order(detect_rainbow_colors(cropped_logo_img,50)))
   
    #cropped_apple_mask = apple_mask[y : y + h, x : x + w]
    #cv.imshow("Wyciente Logo Apple (Maska Binarana)", cropped_apple_mask)

    # momenty geo
    # tu tez wywalic cv musze
    print("\n--- Obliczanie momentów geometrycznych dla maski logo Apple ---")


    
    # WZORCOWE HUE DLA LOGA APPLE:
    #    dla kilku wzorców sprawdzałem
    #    roznia sie nieznacznie
    #    Hu Moment 1: 3.183828
    #    Hu Moment 2: 8.682174
    #    Hu Moment 3: 11.403120
    #    Hu Moment 4: 12.441260
    #    Hu Moment 5: -24.914618
    #    Hu Moment 6: -17.622876
    #    Hu Moment 7: -24.381323

    # a to dane z testowanego obrazu:
    # zauwazylem ze wszystkie sa podobne oprocz momentu 6, on odbiegal czesto
    #   Hu Moment 1: 3.174371
    #   Hu Moment 2: 8.348763
    #   Hu Moment 3: 11.545789
    #   Hu Moment 4: 12.592994
    #   Hu Moment 5: -24.785833
    #   Hu Moment 6: 18.149145
    #   Hu Moment 7: -24.843830 
    desiredHuMoments_Apple =[3.18,8.68,11.4,12.44,-24.91,-17.62,-24.38]
    #   1.5 zapasu w dwie strony (+-) od wzorcowych danych
    HuMomentThreshold=1.5
    correctMoments=0
    print("\nMomenty Inwariantne Hu dla logo Apple:")
    for i in range(0, 7):
        huMoment = to_log_hu(calcHuMoment(apple_mask,i))
        # Normalizacja do skali logarytmicznej dla lepszej porównywalności
        # np.copysign(1.0, x) zwraca znak liczby x, np.log10(abs(x)) zwraca logarytm
        
        if(huMoment<=desiredHuMoments_Apple[i]+HuMomentThreshold and huMoment>=desiredHuMoments_Apple[i]-HuMomentThreshold):
            print(f"Hu moment{i+1} sie zgadza z wzorcem ")
            correctMoments=correctMoments+1
        print(f"  Hu Moment {i+1}: {huMoment:.6f}")
    if(correctMoments>=5):
        print("Momenty hue sie zgadzają, wykryto logo apple!")
        if(is_rainbow_in_order(detect_rainbow_colors(cropped_logo_img,50))):
            img_with_bbox = img.copy()
            # tak samo tutaj
            my_rectangle(img_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imshow("Obraz z ramka logo Apple", img_with_bbox)
            print("Kolory teczy tez sa w odpowiedniej kolejnosci!")

else:
    print("Nie znaleziono dominującego obiektu w kolorach tęczy, który mógłby być logo Apple.")

plt.axis("off")
plt.tight_layout()
plt.show()
cv.waitKey(0)
