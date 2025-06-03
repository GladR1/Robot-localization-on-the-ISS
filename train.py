from ultralytics import YOLO

def train_yolo_pose():

    model = YOLO('yolo11n-pose.pt')

    results = model.train(
        data=r'./robot_stand.yaml',
        epochs=50,
        imgsz=640,
        batch=-1
    )
    
    model.save('yolo11n-pose_trained.pt')
    print("Тренировка завершена!")

if __name__ == '__main__':
    train_yolo_pose()