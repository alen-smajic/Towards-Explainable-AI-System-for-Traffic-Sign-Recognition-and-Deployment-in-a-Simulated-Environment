using UnityEngine.Rendering.PostProcessing;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using System.Linq;

/// <summary>
/// This class controlls all UI elements and makes changes to 
/// the ingame parameters during runtime.
/// </summary>
public class UI : MonoBehaviour
{
    [Header("UI Objects")]
    public GameObject mainMenu; 
    public GameObject ingameMenu;
    public GameObject screenshotOptionsMenu;
    public GameObject signSelectionMenu;
    public GameObject signFreqSlider; 
    public GameObject doubleSignFreqSlider; 
    public GameObject varianceFreqSlider;
    public GameObject occlusionFreqSlider;
    public GameObject autonomousDrivingCheckbox; 
    public GameObject manualDrivingCheckbox;
    public GameObject generateDataCheckbox;
    public GameObject targetPathInputField;
    public GameObject widthInputField;
    public GameObject heightInputField;
    public GameObject captureRateInputField;
    public Dropdown wheatherDropdown; 
    public GameObject rainIntensitySlider; 
    public GameObject rainSliderDisabler; 
    public GameObject lowBeamButton; 
    public GameObject longBeamButton; 
    public GameObject lowBeamSymbol; 
    public GameObject longBeamSymbol; 
    public GameObject turnLeftButton; 
    public GameObject turnRightButton; 
    public GameObject detectionImageDisplay;

    [Header("Camera Mode Icons")]
    public List<GameObject> modeIcons; 
    private int modeID = 0; 

    [Header("Weather Icons")]
    public Sprite sunnyIcon; 
    public Sprite rainyIcon; 
    public Sprite sunriseIcon; 
    public Sprite brightNightIcon; 
    public Sprite darkNightIcon; 

    [Header("Skyboxes")]
    public Material sunnyDay; 
    public Material rainyDay; 
    public Material sunrise; 
    public Material brightNight; 
    public Material darkNight; 

    [Header("Lighting Gameobjects")]
    public GameObject sun;
    public GameObject postProcessVolume;
    public GameObject lowBeamL; 
    public GameObject lowBeamR; 
    public GameObject longBeam;  
    public GameObject longBeamL; 
    public GameObject longBeamR; 
    public GameObject brakeLightL;
    public GameObject brakeLightR;
    public GameObject brakeLightM;
    public GameObject activeBrakeLightL;
    public GameObject activeBrakeLightR;
    public GameObject turningLightL; 
    public GameObject turningLightR; 
    public GameObject interiorLight1;
    public GameObject interiorLight2;
    public GameObject streetLights;

    [Header("Weather Gameobjects")]
    public GameObject rainObject; 

    [Header("Other Gameobjects")]
    public GameObject aiselCar; 
    public GameObject easterEgg;
    public GameObject cascadeClassifier;
    public GameObject datasetGenerator;

    public static List<int> selectedSigns = Enumerable.Range(0, 43).ToList(); 
    public static bool simulationActive = false;
    public static bool generateData = false;
    public static string storeDatasetPath = "";
    public static string widthValue = "1920";
    public static string heightValue = "1080";
    public static string captureRate = "1";
    public static int weatherMode = 0;

    private bool autonomousDriving = true;
    private bool turningRight = false; 
    private bool turningLeft = false;

    // Start is called before the first frame update
    private void Start()
    {
        selectedSigns = Enumerable.Range(0, 43).ToList();
        simulationActive = false;
        generateData = false;
        storeDatasetPath = "";
        widthValue = "1920";
        heightValue = "1080";
        captureRate = "1";
        weatherMode = 0;
        autonomousDriving = true;
        turningRight = false; 
        turningLeft = false; 
        modeID = 0;
        Time.timeScale = 1;
    }

    // Update is called once per frame
    private void Update()
    {
        CheckCameraMode();
        ActivateEasterEgg();
    }

    /// <summary>
    /// Changes the camera mode icons if the user hits the C key.
    /// </summary>
    private void CheckCameraMode()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            modeIcons[modeID].SetActive(false);
            modeID = (modeID + 1) % 5;
            modeIcons[modeID].SetActive(true);

            if (modeID == 4)
            {
                // In camera mode 4, the cascade classifier is activated to detect 
                // traffic signs
                detectionImageDisplay.SetActive(true);
                cascadeClassifier.SetActive(true);
            }
            else
            {
                // In all other modes deactivate the classifier to increase FPS
                // performance
                detectionImageDisplay.SetActive(false);
                cascadeClassifier.SetActive(false);
            }
        }
    }

    /// <summary>
    /// Activates an easter egg which is an 'inside joke' between the developers :)
    /// Gonna leave it here.
    /// </summary>
    private void ActivateEasterEgg()
    {
        if ((Input.GetKey(KeyCode.Keypad0)) && (Input.GetKey(KeyCode.Keypad6)) && (Input.GetKey(KeyCode.Keypad9)))
        {
            easterEgg.SetActive(true);
        }
    }

    /// <summary>
    /// Controlls the slider for traffic sign spawn frequencies.
    /// </summary>
    public void OnSignFreqSliderChanged()
    {
        TrafficSignSpawner.signFreq = (int)Mathf.Ceil(signFreqSlider.GetComponent<Slider>().value * 100);
    }

    /// <summary>
    /// Controlls the slider for double sign frequencies.
    /// </summary>
    public void OnDoubleSignSliderChanged()
    {
        TrafficSignSpawner.doubleSignFreq = (int)Mathf.Ceil(doubleSignFreqSlider.GetComponent<Slider>().value * 100);
    }

    /// <summary>
    /// Controlls the slider for rotation variance frequencies.
    /// </summary>
    public void OnRotationVarianceSliderChanged()
    {
        TrafficSignSpawner.rotationVarianceFreq = (int)Mathf.Ceil(varianceFreqSlider.GetComponent<Slider>().value * 100);
    }

    /// <summary>
    /// Controlls the slider for occlusion frequencies.
    /// </summary>
    public void OnOcclusinSliderChanged()
    {
        TrafficSignSpawner.occlusionFreq = (int)Mathf.Ceil(occlusionFreqSlider.GetComponent<Slider>().value * 100);
    }

    /// <summary>
    /// Controlls the driving mode checkbox inside the main menu.
    /// </summary>
    public void ChangeDrivingMode()
    {
        autonomousDriving = !autonomousDriving;

        if (autonomousDriving)
        {
            // Switches the checkboxes inside the UI
            manualDrivingCheckbox.GetComponent<Toggle>().isOn = false;
            manualDrivingCheckbox.GetComponent<Toggle>().interactable = true;
            manualDrivingCheckbox.transform.GetChild(0).transform.GetChild(0).transform.gameObject.SetActive(false);
            autonomousDrivingCheckbox.GetComponent<Toggle>().interactable = false;
            // Activates autonomous driving inside the car
            aiselCar.GetComponent<CarControllerAI>().enabled = true;
            aiselCar.GetComponent<CarControllerManual>().enabled = false;
        }
        else
        {
            // Switches the checkboxes inside the UI
            autonomousDrivingCheckbox.GetComponent<Toggle>().isOn = false;
            autonomousDrivingCheckbox.GetComponent<Toggle>().interactable = true;
            manualDrivingCheckbox.transform.GetChild(0).transform.GetChild(0).transform.gameObject.SetActive(true);
            manualDrivingCheckbox.GetComponent<Toggle>().interactable = false;
            // Activates manual driving inside the car
            aiselCar.GetComponent<CarControllerAI>().enabled = false;
            aiselCar.GetComponent<CarControllerManual>().enabled = true;
        }
    }

    /// <summary>
    /// Controlls the generate data checkbox inside the main menu.
    /// </summary>
    public void GenerateDataset()
    {
        generateData = !generateData;
        targetPathInputField.GetComponent<InputField>().interactable = generateData;
        screenshotOptionsMenu.SetActive(generateData);
        // Resets all parameters
        storeDatasetPath = "";
        widthInputField.GetComponent<InputField>().text = "";
        heightInputField.GetComponent<InputField>().text = "";
        captureRateInputField.GetComponent<InputField>().text = "";
    }

    /// <summary>
    /// Controlls the target data path input field inside the main menu.
    /// </summary>
    public void OnPathInputChanged()
    {
        storeDatasetPath = targetPathInputField.transform.GetChild(2).GetComponent<Text>().text;
    }

    /// <summary>
    /// Controlls the image width input field inside the main menu.
    /// </summary>
    public void OnWidthInputChanged()
    {
        widthValue = widthInputField.transform.GetChild(2).GetComponent<Text>().text;
    }

    /// <summary>
    /// Controlls the image height input field inside the main menu.
    /// </summary>
    public void OnHeightInputChanged()
    {
        heightValue = heightInputField.transform.GetChild(2).GetComponent<Text>().text;
    }

    /// <summary>
    /// Controlls the screenshot capture rate input field inside the main menu.
    /// </summary>
    public void OnCaptureRateInputChanged()
    {
        captureRate = captureRateInputField.transform.GetChild(2).GetComponent<Text>().text;
    }

    /// <summary>
    /// Controlls the "Press to Start" button inside the main menu.
    /// </summary>
    public void StartSimulation()
    {
        simulationActive = true;
        mainMenu.SetActive(false);
        ingameMenu.SetActive(true);
        datasetGenerator.SetActive(generateData);
    }

    /// <summary>
    /// Controlls the "Select Signs" button inside the main menu.
    /// </summary>
    public void SelectSigns()
    {
        signSelectionMenu.SetActive(true);
    }

    /// <summary>
    /// Manages the list of traffic signs which were selected by the user.
    /// The selection is stored inside the selectedSigns list.
    /// </summary>
    /// <param name="signOption">Gameobject of the selected icon</param>
    public void SelectSign(GameObject signOption)
    {
        // Sign is being deselected
        if (signOption.GetComponent<Image>().color == Color.white)
        {
            signOption.GetComponent<Image>().color = Color.grey;
            int signID;
            int.TryParse(signOption.name, out signID);
            selectedSigns.Remove(signID);
        }
        // Sign is being selected
        else
        {
            signOption.GetComponent<Image>().color = Color.white;
            int signID;
            int.TryParse(signOption.name, out signID);
            selectedSigns.Add(signID);
            selectedSigns.Sort();
        }
    }

    /// <summary>
    /// Controlls the changes inside of the weather dropdown.
    /// </summary>
    /// <param name="dropDown">Dropdown element from the dropdown object</param>
    public void OnDropDownChanged(Dropdown dropDown)
    {
        switch (dropDown.value)
        {
            case 0:
                ChangeEnvironment(sunnyIcon, sunnyDay, true, 0.005f, 10f, 50f, 100f, 
                    3f, 2f, 10f, 2f, 5f, activateStreetLights: false);
                weatherMode = 0;
                break;
            case 1:
                ChangeEnvironment(rainyIcon, rainyDay, false, 0.005f, 10f, 50f, 100f, 
                    3f, 1f, 10f, 1f, 2f, disableRain: false, activateRain: true);
                weatherMode = 1;
                break;
            case 2:
                ChangeEnvironment(sunnyIcon, sunrise, true, 0.005f, 5f, 50f, 100f, 
                    3f, 2f, 10f, 1f, 2f, sunIntensity: 0.82f);
                weatherMode = 2;
                break;
            case 3:
                ChangeEnvironment(brightNightIcon, brightNight, false, 0.07f, 15f, 75f, 150f, 
                    5f, 5f, 25f, 5f, 5f, bloomIntensity: 1f);
                weatherMode = 3;
                break;
            case 4:
                ChangeEnvironment(darkNightIcon, darkNight, false, 0.2f, 100f, 100f, 200f, 
                    30f, 30f, 100f, 15f, 50f, lowBeamIntensity: 15f);
                weatherMode = 4;
                break;
        }
    }

    /// <summary>
    /// Controlls the rain itensity slider inside the ingame menu.
    /// </summary>
    public void OnRainSliderChanged()
    {
        DigitalRuby.RainMaker.BaseRainScript.RainIntensity = rainIntensitySlider.GetComponent<Slider>().value;
    }

    /// <summary>
    /// Changes environment specifications based on its parameters.
    /// Used for the different daytime, lighting and weather conditions.
    /// </summary>
    private void ChangeEnvironment(
        Sprite weatherIcon, Material skybox, bool activateSun, float postProcessWeight, float streetLightsIntensity, 
        float lowBeamRange, float longBeamRange, float longBeamIntensity, float brakeLightIntensity, 
        float activeBrakeLightIntensity, float interioLightIntensity1, float interiorLightIntensity2,
        bool disableRain = true, bool activateRain = false, float sunIntensity = 2.91f, 
        float bloomIntensity = 5f, bool activateStreetLights = true, float lowBeamIntensity = 3f)
    {
        // Environment Settings
        RenderSettings.skybox = skybox;
        rainObject.SetActive(activateRain);
        sun.SetActive(activateSun);
        sun.GetComponent<Light>().intensity = sunIntensity;
        postProcessVolume.GetComponent<PostProcessVolume>().weight = postProcessWeight;
        Bloom bloom = postProcessVolume.GetComponent<PostProcessVolume>().profile.GetSetting<Bloom>();
        bloom.intensity.value = bloomIntensity;
        streetLights.SetActive(activateStreetLights);
        
        // Activates all street lamps inside the scene
        for (int i = 0; i < streetLights.transform.childCount; i++)
        {
            streetLights.transform.GetChild(i).transform.GetComponent<Light>().intensity = streetLightsIntensity;
        }
        DigitalRuby.RainMaker.BaseRainScript.RainIntensity = 0;

        // Car Lighting Settings
        lowBeamL.GetComponent<Light>().range = lowBeamRange;
        lowBeamR.GetComponent<Light>().range = lowBeamRange;
        lowBeamL.GetComponent<Light>().intensity = lowBeamIntensity;
        lowBeamR.GetComponent<Light>().intensity = lowBeamIntensity;
        longBeamL.GetComponent<Light>().range = longBeamRange;
        longBeamR.GetComponent<Light>().range = longBeamRange;
        longBeamL.GetComponent<Light>().intensity = longBeamIntensity;
        longBeamR.GetComponent<Light>().intensity = longBeamIntensity;
        brakeLightL.GetComponent<Light>().intensity = brakeLightIntensity;
        brakeLightR.GetComponent<Light>().intensity = brakeLightIntensity;
        brakeLightM.GetComponent<Light>().intensity = brakeLightIntensity;
        activeBrakeLightL.GetComponent<Light>().intensity = activeBrakeLightIntensity;
        activeBrakeLightR.GetComponent<Light>().intensity = activeBrakeLightIntensity;
        interiorLight1.GetComponent<Light>().intensity = interioLightIntensity1;
        interiorLight2.GetComponent<Light>().intensity = interiorLightIntensity2;

        // UI Settings
        wheatherDropdown.GetComponent<Image>().sprite = weatherIcon;
        rainSliderDisabler.SetActive(disableRain);
        rainIntensitySlider.GetComponent<Slider>().value = 0;
    }

    /// <summary>
    /// Controlls the "low beam" button inside the ingame menu.
    /// </summary>
    public void ActivateLowBeam()
    {
        longBeam.SetActive(false);

        lowBeamButton.SetActive(false);
        longBeamButton.SetActive(true);

        lowBeamSymbol.SetActive(true);
        longBeamSymbol.SetActive(false);
    }

    /// <summary>
    /// Controlls the "long beam" button inside the ingame menu.
    /// </summary>
    public void ActivateLongBeam()
    {
        longBeam.SetActive(true);

        longBeamButton.SetActive(false);
        lowBeamButton.SetActive(true);

        longBeamSymbol.SetActive(true);
        lowBeamSymbol.SetActive(false);
    }

    /// <summary>
    /// Controlls the "turn right" button inside the ingame menu.
    /// </summary>
    public void TurnRight()
    {
        if (turningRight)
        {
            turningRight = false; 
            StopAllCoroutines(); 
            turningLightR.SetActive(false); 
            turnRightButton.GetComponent<Image>().color = Color.white; 
        }
        else
        {
            turningRight = true; 
            turningLeft = false; 
            StopAllCoroutines(); 
            turningLightL.SetActive(false); 
            turnLeftButton.GetComponent<Image>().color = Color.white; 
            StartCoroutine(ActivateTurnLight("right")); 
        }
    }

    /// <summary>
    /// Controlls the "turn left" button inside the ingame menu.
    /// </summary>
    public void TurnLeft()
    {
        if (turningLeft)
        {
            turningLeft = false; 
            StopAllCoroutines(); 
            turningLightL.SetActive(false); 
            turnLeftButton.GetComponent<Image>().color = Color.white; 
        }
        else
        {
            turningLeft = true; 
            turningRight = false; 
            StopAllCoroutines(); 
            turningLightR.SetActive(false); 
            turnRightButton.GetComponent<Image>().color = Color.white; 
            StartCoroutine(ActivateTurnLight("left")); 
        }
    }

    /// <summary>
    /// Activates the turn lights depending on the direction parameter. Makes them
    /// go on and off every .4 seconds.
    /// </summary>
    /// <param name="direction">String with either "left" or "right" value</param>
    /// <returns></returns>
    IEnumerator ActivateTurnLight(string direction)
    {
        while (true)
        {
            if (direction == "left")
            {
                turningLightL.SetActive(true);
                turnLeftButton.GetComponent<Image>().color = Color.yellow;
                yield return new WaitForSeconds(.4f);
                turningLightL.SetActive(false);
                turnLeftButton.GetComponent<Image>().color = Color.white;
                yield return new WaitForSeconds(.4f);
            }
            else
            {
                turningLightR.SetActive(true);
                turnRightButton.GetComponent<Image>().color = Color.yellow;
                yield return new WaitForSeconds(.4f);
                turningLightR.SetActive(false);
                turnRightButton.GetComponent<Image>().color = Color.white;
                yield return new WaitForSeconds(.4f);
            }
        }
    }

    /// <summary>
    /// Controlls the "reset" button inside the ingame menu.
    /// </summary>
    public void ResetScene()
    {
        simulationActive = false;
        SceneManager.LoadScene(0);
    }

    /// <summary>
    /// Controlls the "exit" button inside the ingame menu.
    /// </summary>
    public void Exit()
    {
        Application.Quit();
    }
}
