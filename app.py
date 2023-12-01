# app.py

from flask import Flask, render_template, send_file
import cv2
import numpy as np
import base64
import zlib
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    path = "static/image.png"
    return render_template('index.html', defaultImg=path)

@app.route('/grayscale')  # Chuyển đổi sang ảnh trắng (grayscale)
def grayscale():
    path = 'static/image2.jpg'
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
    path = 'static/image2.jpg'
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
    path = 'static/image2.jpg'
    # Đọc ảnh từ file hoặc camera
    image = cv2.imread('static/image2.jpg')

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
                           kernel="3x3", purpose=purpose, defaultImg=path)


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

def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

@app.route('/prewitt')  # Bộ lọc Prewitt
def prewitt():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng bộ lọc Prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewittx = cv2.filter2D(image, -1, kernelx)
    prewitty = cv2.filter2D(image, -1, kernely)

    # Kết hợp hai ảnh
    prewitt_image = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    img_str = encode_image(prewitt_image)

    purpose = """Bộ lọc Prewitt được sử dụng để phát hiện biên cạnh trong ảnh. Bộ lọc Prewitt giúp tạo ra một ảnh mới mà ở đó các biên cạnh (nơi mà mức độ cường độ ảnh thay đổi đột ngột) được làm nổi bật. Điều này rất hữu ích trong nhiều ứng dụng, bao gồm phân loại hình ảnh, phát hiện đối tượng, và theo dõi chuyển động."""

    return render_template('index.html', img_data=img_str, title="Bộ lọc Prewitt", algorithm="Bộ lọc Prewitt",
                           purpose=purpose, defaultImg=path)

@app.route('/negative')  # Biến đổi âm bản
def negative():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng biến đổi âm bản
    negative_image = cv2.bitwise_not(image)

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    img_str = encode_image(negative_image)

    purpose = """Áp dụng biến đổi âm bản để tạo ra một ảnh mới mà trong đó mức độ cường độ của mỗi điểm ảnh đã được đảo ngược (các điểm sáng trở thành tối và ngược lại)."""

    return render_template('index.html', img_data=img_str, title="Biến đổi âm bản", algorithm="Biến đổi âm bản",
                            purpose=purpose, defaultImg=path)

@app.route('/threshold')  # Biến đổi ngưỡng Threshold
def threshold():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng biến đổi ngưỡng
    ret, threshold_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    img_str = encode_image(threshold_image)

    purpose = """Áp dụng biến đổi ngưỡng để tạo ra một ảnh nhị phân, trong đó các điểm ảnh có giá trị lớn hơn hoặc bằng ngưỡng được đặt thành một giá trị (thường là màu trắng), và các điểm ảnh có giá trị nhỏ hơn ngưỡng được đặt thành một giá trị khác (thường là màu đen)."""

    return render_template('index.html', img_data=img_str, title="Biến đổi ngưỡng Threshold", algorithm="Biến đổi ngưỡng Threshold",
                            purpose=purpose, defaultImg=path)

@app.route('/sobel')  # Thuật toán Sobel
def sobel():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng thuật toán Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Kết hợp Sobel x và Sobel y để có kết quả cuối cùng
    sobel_image = cv2.magnitude(sobel_x, sobel_y)

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    img_str = encode_image(sobel_image)

    purpose = """Áp dụng thuật toán Sobel để phát hiện cạnh trong ảnh. Sobel xác định đạo hàm riêng của ảnh theo hướng ngang và dọc, sau đó kết hợp chúng để tìm cạnh."""

    return render_template('index.html', img_data=img_str, title="Thuật toán Sobel", algorithm="Thuật toán Sobel",
                           purpose=purpose, defaultImg=path)

@app.route('/histogram')  # Cân bằng Histogram
def histogram():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Tính histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Tính hàm phân phối tích lũy (CDF) của histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Thực hiện cân bằng histogram
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)

    # Định lại hình dạng của ảnh cân bằng trở lại hình dạng ban đầu
    equalized_image = equalized_image.reshape(image.shape)

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    img_str = encode_image(equalized_image)

    purpose = """Áp dụng cân bằng histogram để cải thiện độ tương phản và chi tiết trong ảnh."""

    return render_template('index.html', img_data=img_str, title="Cân bằng Histogram", algorithm="Cân bằng Histogram",
                            purpose=purpose, defaultImg=path)

def weighted_average_filter(image, weights):
    # Áp dụng trung bình có trọng số
    weighted_avg_image = cv2.addWeighted(image, weights[0], image, weights[1], weights[2])
    return weighted_avg_image

@app.route('/weighted_average')  # Trung bình có trọng số
def weighted_average():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Định nghĩa trọng số cho trung bình có trọng số
    weights = [0.5, 0.5, 0]

    # Áp dụng biến đổi trung bình có trọng số
    weighted_avg_image = weighted_average_filter(image, weights)

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    img_str = encode_image(weighted_avg_image)

    purpose = """Áp dụng thuật toán trung bình có trọng số để tạo ra một sự kết hợp của ảnh gốc và ảnh được chỉnh sửa với các trọng số xác định."""

    return render_template('index.html', img_data=img_str, title="Trung bình có trọng số", algorithm="Trung bình có trọng số",
                            purpose=purpose, defaultImg=path)

##############
# Bộ lọc KNN
def knn_filtered(image, k=5):
    knn_filter = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=k)
    return knn_filter

@app.route('/knn_average')  # Biến đổi KNN Average
def knn_average():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng biến đổi KNN Average với giá trị k cụ thể
    knn_filtered_image = knn_filtered(image, k=5)

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', knn_filtered_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Áp dụng thuật toán KNN (K-Nearest Neighbors) để làm trơn ảnh và giảm nhiễu."""

    return render_template('index.html', img_data=img_str, title="KNN Average", algorithm="KNN Average",
                            purpose=purpose, defaultImg=path)

@app.route('/roberts')
def roberts():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Định nghĩa kernel Roberts Cross
    roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_kernel_y = np.array([[0, -1], [1, 0]], dtype=np.float32)

    # Áp dụng biến đổi Roberts Cross
    roberts_image_x = cv2.filter2D(image, cv2.CV_64F, roberts_kernel_x)
    roberts_image_y = cv2.filter2D(image, cv2.CV_64F, roberts_kernel_y)
    roberts_image = cv2.magnitude(roberts_image_x, roberts_image_y)
    roberts_image = cv2.convertScaleAbs(roberts_image)

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', roberts_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Sử dụng toán tử Roberts để phát hiện biên ảnh. Toán tử này nhấn mạnh sự thay đổi giữa các pixel liền kề theo chiều ngang và chiều dọc."""

    return render_template('index.html', img_data=img_str, title="Biến đổi Roberts Cross", algorithm="Biến đổi Roberts Cross",
                            purpose=purpose, defaultImg=path)

@app.route('/canny')
def canny():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng phát hiện biên Canny
    canny_image = cv2.Canny(image, 50, 150)  # Bạn có thể điều chỉnh giá trị ngưỡng theo cần thiết

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', canny_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Sử dụng thuật toán Canny để phát hiện biên ảnh. Canny được sử dụng để tìm ra các biên tuyến tính trong ảnh."""

    return render_template('index.html', img_data=img_str, title="Biến đổi Canny", algorithm="Biến đổi Canny",
                            purpose=purpose, defaultImg=path)

@app.route('/otsu')
def otsu():
    path = 'static/image2.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng ngưỡng Otsu
    _, otsu_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Chuyển ảnh sang định dạng base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', otsu_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Áp dụng thuật toán ngưỡng Otsu để tạo ra một ảnh nhị phân. Thuật toán này tự động tìm ra ngưỡng phân loại tốt nhất cho ảnh."""

    return render_template('index.html', img_data=img_str, title="Biến đổi Otsu", algorithm="Biến đổi Otsu",
                            purpose=purpose, defaultImg=path)

def dilation_transform(image):
    # Định nghĩa kernel cho dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Áp dụng thuật toán dilation
    dilated_image = cv2.dilate(image, kernel, iterations=1)

    return dilated_image

@app.route('/dilation')  # Biến đổi dilation
def dilation():
    path = 'static/image2.jpg'
    # Đọc ảnh từ file hoặc camera
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng thuật toán dilation
    dilated_image = dilation_transform(image)

    # Chuyển đổi ảnh thành base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', dilated_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Áp dụng thuật toán dilation để mở rộng vùng đối tượng trong ảnh."""

    return render_template('index.html', img_data=img_str, title="Biến đổi dilation", algorithm="Biến đổi dilation",
                            purpose=purpose, defaultImg=path)


def opening_transform(image):
    # Định nghĩa kernel cho opening
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Áp dụng thuật toán opening
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return opened_image

@app.route('/opening')  # Biến đổi Opening
def opening():
    path = 'static/image2.jpg'
    # Đọc ảnh từ file hoặc camera
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng thuật toán opening
    opened_image = opening_transform(image)

    # Chuyển đổi ảnh thành base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', opened_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Áp dụng thuật toán opening để loại bỏ nhiễu và mở rộng vùng đối tượng trong ảnh."""

    return render_template('index.html', img_data=img_str, title="Biến đổi Opening", algorithm="Biến đổi Opening",
                            purpose=purpose, defaultImg=path)

def closing_transform(image):
    # Định nghĩa kernel cho closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Áp dụng thuật toán closing
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return closed_image

@app.route('/closing')  # Biến đổi Closing
def closing():
    path = 'static/image2.jpg'
    # Đọc ảnh từ file hoặc camera
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Áp dụng thuật toán closing
    closed_image = closing_transform(image)

    # Chuyển đổi ảnh thành base64 để truyền qua HTML
    _, buffer = cv2.imencode('.png', closed_image)
    img_str = base64.b64encode(buffer).decode('utf-8')

    purpose = """Áp dụng thuật toán closing để đóng kín vùng đối tượng trong ảnh và loại bỏ những lỗ nhỏ."""

    return render_template('index.html', img_data=img_str, title="Biến đổi Closing", algorithm="Biến đổi Closing",
                            purpose=purpose, defaultImg=path)


if __name__ == '__main__':
    app.run(debug=True)
