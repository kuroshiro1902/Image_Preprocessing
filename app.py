# app.py

from flask import Flask, render_template, send_file
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    path = "static/image.png"
    return render_template('index.html', defaultImg=path)

@app.route('/grayscale')  # Chuyển đổi sang ảnh trắng (grayscale)
def grayscale():
    path = 'static/BW.jpg'
     # Đọc ảnh từ file hoặc camera
    image = cv2.imread(path)

    # Chuyển ảnh sang trắng
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Chuyển ảnh sang dạng base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', gray_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Chuyển đổi ảnh sang ảnh xám (grayscale), giảm chiều sâu màu và giữ lại chi tiết cấu trúc."""

    return render_template('index.html', img_data=img_str, title="Lọc xám", algorithm="Lọc xám (Grayscale)",
                           kernel="null", purpose=purpose, defaultImg=path)

@app.route('/mean_filtering') #Lọc trung bình
def mean_filtering():
    path = 'static/image.jpg'
     # Đọc ảnh từ file hoặc camera
    image = cv2.imread(path)

    # Kích thước kernel (phần tử lọc)
    kernel_size = (3,3)

    # Thực hiện lọc trung bình
    blurred_image = cv2.blur(image, kernel_size)

    # Chuyển ảnh sang dạng base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', blurred_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Khử nhiễu bằng cách làm mờ ảnh, các hạt nhiễu có kích thước nhỏ trong ảnh sẽ bị làm mờ đi."""

    return render_template('index.html', img_data=img_str, title="Lọc trung bình", algorithm="Lọc trung bình (Mean-filtering - Blur)",
                           kernel="3x3", purpose=purpose, defaultImg=path)

@app.route('/median_filtering')  # Lọc trung vị
def median_filtering():
    # Đọc ảnh từ file hoặc camera
    image = cv2.imread('static/image.jpg')

    # Kích thước kernel (phần tử lọc)
    kernel_size = 3

    # Thực hiện lọc trung vị
    median_blurred_image = cv2.medianBlur(image, kernel_size)

    # Chuyển ảnh sang dạng base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', median_blurred_image)

    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Bộ lọc trung vị được sử dụng tương đối phổ biến trong xử lý ảnh để loại bỏ nhiễu 
vì nhiễu thường có giá trị bất thường so với các mức xám của ảnh gốc. Sau khi qua bộ
lọc trung vị, các giá trị bất thường sẽ bị loại bỏ trong khi đó ảnh không bị mờ giống như 
bộ lọc trung bình. Chính vì vậy, bộ lọc trung vị thường dùng để loại bỏ nhiễu xung (hay 
còn gọi là nhiễu muối tiêu) vì các nhiễu này xuất hiện dưới dạng các chấm trắng hoặc 
đen nổi bật trong ảnh."""

    return render_template('index.html', img_data=img_str, title="Lọc trung vị", algorithm="Lọc trung vị (Median-filtering)",
                           kernel="3x3", purpose=purpose, defaultImg="static/image.jpg")


@app.route('/gaussian')  # Bộ lọc Gaussian
def gaussian():
    path = 'static/image2.jpg'
    # Đọc ảnh từ file hoặc camera
    image = cv2.imread(path)

    # Kích thước kernel (phần tử lọc)
    kernel_size = (5, 5)

    # Độ chệch chuẩn (sigma) của Gaussian
    sigma = 0

    # Thực hiện lọc Gaussian
    gaussian_blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    # Chuyển ảnh sang dạng base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', gaussian_blurred_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Bộ lọc Gaussian được sử dụng để làm mờ ảnh và giảm nhiễu. Nó hoạt động bằng cách tính trung bình trọng số của các pixel xung quanh mỗi pixel trong ảnh, với trọng số giảm dần theo khoảng cách từ pixel cần làm mờ. Bộ lọc Gaussian có thể giúp giảm nhiễu mà vẫn giữ được chi tiết cạnh trong ảnh.

Tham số Sigma là độ chệch chuẩn của Gaussian, ảnh hưởng đến độ mịn của quá trình làm mờ."""

    return render_template('index.html', img_data=img_str, title="Bộ lọc Gaussian", algorithm="Bộ lọc Gaussian (Gaussian Blur)",
                           kernel="5x5", purpose=purpose, defaultImg=path)

@app.route('/laplacian')  # Bộ lọc Laplacian
def laplacian():
    path = 'static/image2.jpg'
    # Đọc ảnh từ file hoặc camera
    image = cv2.imread(path)

    # Áp dụng bộ lọc Laplacian
    laplacian_image = cv2.Laplacian(image, cv2.CV_64F)

    # Chuyển ảnh sang dạng base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', laplacian_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Áp dụng bộ lọc Laplacian để kiểm tra biên cạnh và cấu trúc trong ảnh (các biên và cạnh là các điểm có mức xám thay đổi đột ngột trong ảnh)."""

    return render_template('index.html', img_data=img_str, title="Bộ lọc Laplacian", algorithm="Bộ lọc Laplacian",
                            purpose=purpose, defaultImg=path)

@app.route('/prewitt')  # Prewitt filter
def prewitt():
    path = 'static/image2.jpg'
    # Read image from file or camera
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Apply Prewitt filter
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewittx = cv2.filter2D(image, -1, kernelx)
    prewitty = cv2.filter2D(image, -1, kernely)

    # Combine the two images
    prewitt_image = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)

    # Convert image to base64 to pass through HTML
    _, buffer = cv2.imencode('.png', prewitt_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Bộ lọc Prewitt được sử dụng để phát hiện biên cạnh trong ảnh. Bộ lọc Prewitt giúp tạo ra một ảnh mới mà ở đó các biên cạnh (nơi mà mức độ cường độ ảnh thay đổi đột ngột) được làm nổi bật. Điều này rất hữu ích trong nhiều ứng dụng, bao gồm phân loại hình ảnh, phát hiện đối tượng, và theo dõi chuyển động."""

    return render_template('index.html', img_data=img_str, title="Bộ lọc Prewitt", algorithm="Bộ lọc Prewitt",
                           purpose=purpose, defaultImg=path)

@app.route('/negative')  # Negative transformation
def negative():
    path = 'static/image2.jpg'
    # Read image from file or camera
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Apply negative transformation
    negative_image = cv2.bitwise_not(image)

    # Convert image to base64 to pass through HTML
    _, buffer = cv2.imencode('.png', negative_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Áp dụng biến đổi âm bản để tạo ra một ảnh mới mà trong đó mức độ cường độ của mỗi điểm ảnh đã được đảo ngược (các điểm sáng trở thành tối và ngược lại)."""

    return render_template('index.html', img_data=img_str, title="Biến đổi âm bản", algorithm="Biến đổi âm bản",
                            purpose=purpose, defaultImg=path)

@app.route('/threshold')  # Threshold transformation
def threshold():
    path = 'static/image2.jpg'
    # Read image from file or camera
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Apply threshold transformation
    ret, threshold_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Convert image to base64 to pass through HTML
    _, buffer = cv2.imencode('.png', threshold_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Áp dụng biến đổi ngưỡng để tạo ra một ảnh nhị phân, trong đó các điểm ảnh có giá trị lớn hơn hoặc bằng ngưỡng được đặt thành một giá trị (thường là màu trắng), và các điểm ảnh có giá trị nhỏ hơn ngưỡng được đặt thành một giá trị khác (thường là màu đen)."""

    return render_template('index.html', img_data=img_str, title="Biến đổi ngưỡng Threshold", algorithm="Biến đổi ngưỡng Threshold",
                            purpose=purpose, defaultImg=path)


if __name__ == '__main__':
    app.run(debug=True)
