using UnityEngine;
using UnityEngine.UI;
using OpenCvSharp;
using System.Collections.Generic;
using Unity.Barracuda;

public class CascadeDetection : MonoBehaviour
{
    public List<TextAsset> features = new List<TextAsset>();
    
    public RenderTexture rTex; // RenderTexture which gets data from the camera
    public GameObject outSurface;  // Displays the detection output on a raw image on the UI

    protected List<CascadeClassifier> cascadeClassifiers = new List<CascadeClassifier>();

    protected List<FileStorage> storageFeatures = new List<FileStorage>();

    private OpenCvSharp.Unity.TextureConversionParams texParams;
    private Texture2D curTexture2D; // Used to hold the current frame during the inference pipeline
    private Mat curTextureMat; // Stores the image matrix of the currrent frame
    private Mat dreck;


    Texture2D tex;

    int test;

    public NNModel modelSource;

    // Start is called before the first frame update
    void Start()
    {
        test = 0;
        for(int i = 0; i < 4; i++)
        {
            cascadeClassifiers.Add(new CascadeClassifier());
            storageFeatures.Add(new FileStorage(features[i].text, FileStorage.Mode.Read | FileStorage.Mode.Memory));
            if (!cascadeClassifiers[i].Read(storageFeatures[i].GetFirstTopLevelNode()))
                throw new System.Exception("FaceProcessor.Initialize: Failed to load faces cascade classifier");
        }

        texParams = new OpenCvSharp.Unity.TextureConversionParams();

        texParams.RotationAngle = 0;
        texParams.FlipHorizontally = false;
        texParams.FlipVertically = false;

        tex = new Texture2D(1920, 1080, TextureFormat.RGB24, false);
    }

    void Update()
    {
        curTexture2D = toTexture2D();

        curTextureMat = OpenCvSharp.Unity.TextureToMat(curTexture2D as UnityEngine.Texture2D, texParams);

        for(int i = 0; i < 4; i++)
        {
            OpenCvSharp.Rect[] detections = cascadeClassifiers[i].DetectMultiScale(curTextureMat, 2.5f, 1);



            for (int j = 0; j < detections.Length; j++)
            {
                Color[] pix = curTexture2D.GetPixels(detections[j].X, detections[j].Y, detections[j].Width, detections[j].Height);
                Texture2D cutTexture2D = new Texture2D(detections[j].Width, detections[j].Height);
                cutTexture2D.SetPixels(pix);
                cutTexture2D.Apply();

                //byte[] bytes = cutTexture2D.EncodeToPNG();
                //string fileName = "C:/Users/alens/Desktop/bestetest/test" + test.ToString() +".png";
                //System.IO.File.WriteAllBytes(fileName, bytes);

                test  = test +  1;


                Cv2.Rectangle((InputOutputArray)curTextureMat, detections[j], Scalar.FromRgb(255, 0, 0), 2);
            }

            
        }

        

        curTexture2D = OpenCvSharp.Unity.MatToTexture(curTextureMat, curTexture2D);

        RenderFrame();
    }

    Texture2D toTexture2D()
    {
        RenderTexture.active = rTex;
        tex.ReadPixels(new UnityEngine.Rect(0, 0, rTex.width, rTex.height), 0, 0);
        tex.Apply();
        return tex;
    }

    private void RenderFrame()
    {
        if (curTexture2D != null)
        {
            // apply
            outSurface.GetComponent<RawImage>().texture = curTexture2D;

            // Adjust image ration according to the texture sizes 
            outSurface.GetComponent<RectTransform>().sizeDelta = new Vector2(curTexture2D.width, curTexture2D.height);
        }
    }


    void classifier()
    {
        var model = ModelLoader.Load(modelSource);
        var worker = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputePrecompiled, model);
    }
}
