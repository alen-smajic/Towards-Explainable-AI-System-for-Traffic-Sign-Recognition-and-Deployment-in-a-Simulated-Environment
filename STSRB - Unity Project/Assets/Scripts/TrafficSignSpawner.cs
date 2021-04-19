using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// This class controlls the spawning of the traffic signs inside the simulation.
/// It uses random distributions to spawn the traffic signs inside the scene.
/// </summary>
public class TrafficSignSpawner : MonoBehaviour
{
    [Header("Traffic Sign Objects")]
    public List<GameObject> trafficPoles = new List<GameObject>();
    public List<GameObject> trafficSignsNormal = new List<GameObject>();
    public List<GameObject> trafficSignsPink = new List<GameObject>();
    public List<Material> signStickerMaterials = new List<Material>();
    public GameObject signStickerPatch;

    [Header("Spawn Frequency")]
    public static int signFreq = 50;
    public static int doubleSignFreq = 25;
    public static int rotationVarianceFreq = 13;
    public static int occlusionFreq = 5;
    public int maxStickerAmount;

    [Header("Parent Sign Objects")]
    public GameObject trafficSignsParentObject;
    public GameObject trafficSignPinkParentObject;

    private bool spawningStarted = false;

    // Start is called before the first frame update
    private void Start()
    {
        signFreq = 50;
        doubleSignFreq = 25;
        rotationVarianceFreq = 13;
        occlusionFreq = 5;

        spawningStarted = false;
    }

    // Update is called once per frame
    private void Update()
    {
        // Starts spawning, when simulation started
        if (UI.simulationActive)
        {
            if(!spawningStarted)
            {
                spawningStarted = true;
                SpawnSigns();
            }
        }
    }

    /// <summary>
    /// Activates a random amount of traffic sign poles from the trafficPoles list and instantiates the traffic
    /// signs on top of these poles, with respect to the frequency parameters.
    /// </summary>
    private void SpawnSigns()
    {
        foreach (GameObject pole in trafficPoles)
        {
            // Each traffic sign pole from the list is activated with probability signFreq
            if (Random.Range(0, 100) < signFreq)
            {
                pole.SetActive(true);
                spawnSingleSign(pole, 2.5f); // Spawns one traffic sign on top of the pole

                // Spanws a second traffic sign onto the pole with probability doubleSignFreq
                if (Random.Range(0, 100) < doubleSignFreq)
                {
                    spawnSingleSign(pole, 1.55f);
                }
            }
        }
    }

    /// <summary>
    /// Instantiates a single traffic sign onto the pole. Applies rotation with respect to the rotationVarianceFreq
    /// parameter. Puts the traffic sign on the correct height.
    /// If the simulation is used to generate data, it spawns also a twin traffic sign at the exact same location
    /// with pink texture.
    /// Furthermore, it generates a random amount of stickers, which are being applied to the traffic sign to simulate
    /// occlusion.
    /// </summary>
    /// <param name="pole">Target traffic pole used for location information</param>
    /// <param name="height">Height value for spawning the traffic sign</param>
    private void spawnSingleSign(GameObject pole, float height)
    {
        float rotationVariance = 0f;

        // Each traffic sign will contain some rotation with probability rotationVarianceFreq
        if (Random.Range(0, 100) < rotationVarianceFreq)
        {
            // The first term is the maximum rotation angle
            // The second term generates a random number to use it as the proportion of the maximum rotation angle
            // The third term generates either a negativ or a positive number, to rotate the traffic sign in either direction
            rotationVariance = 30 * ((float)Random.Range(0, 101) / 100) * (Random.Range(0, 2) * 2 - 1);
        }

        // Picks a random traffic sign from the list of active traffic signs and instantiates it at the top of the pole
        // Rotates the traffic sign in the direction of the traffic pole and applies the rotation variance
        int sign_idx = UI.selectedSigns[Random.Range(0, UI.selectedSigns.Count)];
        GameObject new_sign = Instantiate(trafficSignsNormal[sign_idx],
            new Vector3(pole.transform.position.x, pole.transform.position.y + height, pole.transform.position.z),
            Quaternion.Euler(0, pole.transform.rotation.eulerAngles.y, rotationVariance));
        new_sign.transform.parent = trafficSignsParentObject.transform; // Assign traffic sign to parent object

        // Each traffic sign contains a sticker patch with probability occlusionFreq
        if (Random.Range(0, 100) < occlusionFreq)
        {
            // Outputs a random amount of patches (at least 1)
            int randAmountPatches = Random.Range(1, maxStickerAmount+1);
            
            // Each patch is placed onto the traffic sign with a random position, rotation and scale value
            for(int i = 0; i < randAmountPatches; i++)
            {
                // Instatiates a random sticker at the correct position
                GameObject new_patch = Instantiate(signStickerPatch,
                    new Vector3(pole.transform.position.x, pole.transform.position.y + height, pole.transform.position.z),
                    Quaternion.Euler(-90, pole.transform.rotation.eulerAngles.y, pole.transform.rotation.eulerAngles.z));
                new_patch.transform.parent = new_sign.transform; // Assign patch sticker to the traffic sign

                // Applys random texture
                new_patch.GetComponent<Renderer>().material = signStickerMaterials[Random.Range(0, signStickerMaterials.Count)];

                // Places the sticker using random X and Y coordinates
                // The Z coordinate is scaled by the index i to prevent texture clipping
                new_patch.transform.localPosition = new Vector3(Random.Range(-0.206f, 0.206f),
                    Random.Range(-0.206f, 0.206f), -0.05f + 0.001f * i);

                // Applys random rotation
                new_patch.transform.Rotate(Vector3.up, Random.Range(0,360));

                // Applys random scale to the sticker
                new_patch.transform.localScale = new Vector3(Random.Range(0.005303489f, 0.01422123f), 
                    Random.Range(0.005303489f, 0.01422123f), Random.Range(0.005303489f, 0.01422123f));
            }
        }

        if (UI.generateData)
        {
            // Instantiates twin traffic sign (at the exact same location) with pink texture for creating the dataset
            GameObject new_pink_sign = Instantiate(trafficSignsPink[sign_idx],
                new Vector3(pole.transform.position.x, pole.transform.position.y + height, pole.transform.position.z),
                Quaternion.Euler(0, pole.transform.rotation.eulerAngles.y, rotationVariance));
            new_pink_sign.transform.parent = trafficSignPinkParentObject.transform; // Assign traffic sign to parent object
        }
    }
}
