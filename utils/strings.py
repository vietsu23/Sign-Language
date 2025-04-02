
class ExpressionHandler:

    MAPPING = {
        "bình_thường": "Ngồi yên",
        "cảm_ơn": "Cảm ơn",
        "xin_chào": "Xin chào",
        "yêu": "Yêu",
        "không": "Không",
        "bạn" : "Bạn",
        "đi_đến" : "Đi đến",
        "gặp_gỡ": "Gặp mặt",
        "hẹn_gặp_lại" :"Hẹn gặp lại",
        "học" :"học",
        "làm_ơn": "Xin hãy (làm ơn)",
        "muốn": "Muốn",
        "tôi": "tôi",
        "giúp": "Giúp đỡ",  
        "ăn":"Ăn",
        "uống":"Uống",
        "bạn_bè":"Bạn bè",
        "cần":"Cần",
        "chơi":"Chơi",
        "gia_đình":"Gia đình",
        "suy_nghĩ":"Suy nghĩ",
        "hơn_nữa":"Nhiều hơn nữa",
        "nhìn":"Nhìn thấy / đọc / xem",
        "sách":"Quyển sách",
        "xong":"Xong (hoàn thành)"
    }

    def __init__(self):
        # Lưu trữ lại thông điệp hiện tại
        self.current_message = ""

    def receive(self, message):
        self.current_message = message

    def get_message(self):
        return ExpressionHandler.MAPPING[self.current_message]
