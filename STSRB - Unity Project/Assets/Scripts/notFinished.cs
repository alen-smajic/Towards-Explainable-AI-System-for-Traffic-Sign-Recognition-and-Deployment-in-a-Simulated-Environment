using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

public class ttestn : MonoBehaviour
{
    public NNModel modelSource;
    public Texture2D texture;


    // Start is called before the first frame update
    void Start()
    {

        var model = ModelLoader.Load(modelSource);
        var worker = BarracudaWorkerFactory.CreateWorker(BarracudaWorkerFactory.Type.ComputePrecompiled, model);
        texture = Resize(texture, 32, 32);
        var tensor = new Tensor(texture, 3);
        worker.Execute(tensor);
        Tensor O = worker.PeekOutput();
        worker.Fetch();
        print(O);
        print(O.ArgSort());
        print(O.ArgMax()[0]);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    Texture2D Resize(Texture2D texture2D, int targetX, int targetY)
    {
        RenderTexture rt = new RenderTexture(targetX, targetY, 24);
        RenderTexture.active = rt;
        Graphics.Blit(texture2D, rt);
        Texture2D result = new Texture2D(targetX, targetY);
        result.ReadPixels(new Rect(0, 0, targetX, targetY), 0, 0);
        result.Apply();
        return result;
    }
}
