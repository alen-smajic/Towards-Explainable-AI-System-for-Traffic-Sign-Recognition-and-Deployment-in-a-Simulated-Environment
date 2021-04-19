using UnityEngine;

/// <summary>
/// This class controlls the car in manual mode. It uses the user input to move the car.
/// </summary>
public class CarControllerManual : MonoBehaviour
{
    [Header("Car Objects")]
    public WheelCollider wheelFL;
    public WheelCollider wheelFR;
    public WheelCollider wheelRL;
    public WheelCollider wheelRR;
    public GameObject brakeLights;
    public GameObject brakeLightsLed;
    public GameObject steeringWheel;

    [Header("Car Specifications")]
    public Vector3 centerOfMass;
    public float maxSteerAngle;
    public float maxMotorTorque;
    public float maxBoostMotorTorque;
    public float maxBreakTorque;
    public float maxSpeed;

    private float currentSpeed = 0f;
    private float maxMotorTorqueCache;
    private bool isBraking = false;
    
    // Start is called before the first frame update
    private void Start()
    {
        // Applies new center of mass to the car
        GetComponent<Rigidbody>().centerOfMass = centerOfMass;

        currentSpeed = 0f;
        maxMotorTorqueCache = maxMotorTorque;
        isBraking = false;
    }

    // Update is called once per frame
    private void FixedUpdate()
    {
        // Enable user input, once the simulation is active
        if (UI.simulationActive)
        {
            Drive();
            ApplySteer();
            Brake();
        }
    }

    /// <summary>
    /// Applies motor torque to all wheels, if the car is not braking and has not 
    /// reached its full speed.
    /// </summary>
    private void Drive()
    {
        // Calculates the speed of the car
        currentSpeed = 2 * Mathf.PI * wheelRL.radius * wheelRL.rpm * 60 / 1000;

        if (!isBraking && currentSpeed < maxMotorTorque)
        {
            // Takes user input, which increases over time from 0 to 1
            float motorTorque = Input.GetAxis("Vertical");

            wheelFL.motorTorque = maxMotorTorque * motorTorque; 
            wheelFR.motorTorque = maxMotorTorque * motorTorque; 
            wheelRL.motorTorque = maxMotorTorque * motorTorque;
            wheelRR.motorTorque = maxMotorTorque * motorTorque;
        }
        else
        {
            wheelFL.motorTorque = 0f;
            wheelFR.motorTorque = 0f;
            wheelRL.motorTorque = 0f;
            wheelRR.motorTorque = 0f;
        }
    }

    /// <summary>
    /// Takes the user input to apply the new steering angle to the steering wheel and 
    /// the tires.
    /// </summary>
    private void ApplySteer()
    {
        // User steering input and steering angle, which increases over time from 0 to 1
        float steeringStrength = Input.GetAxis("Horizontal");
        float newSteer = steeringStrength * maxSteerAngle;

        // Rotates the car tires
        wheelFL.steerAngle = newSteer;
        wheelFR.steerAngle = newSteer;

        // Rotates the steering wheel
        Quaternion nextRotation = Quaternion.Lerp(steeringWheel.transform.localRotation, 
            Quaternion.Euler(16.643f, 0f, -newSteer * 3), Time.deltaTime * 3f);
        steeringWheel.transform.localRotation = nextRotation;
    }

    /// <summary>
    /// Applies braking to all tires once the user hits space.
    /// </summary>
    private void Brake()
    {
        // Applies the braking to the tires and activates the braking lights
        if (Input.GetKey(KeyCode.Space))
        {
            wheelFL.brakeTorque = maxBreakTorque;
            wheelFR.brakeTorque = maxBreakTorque;
            wheelRL.brakeTorque = maxBreakTorque;
            wheelRR.brakeTorque = maxBreakTorque;

            brakeLights.SetActive(true);
            brakeLightsLed.SetActive(true);

            isBraking = true;
        }
        // Removes the braking effect and deactivates the braking lights
        else
        {
            wheelFL.brakeTorque = 0f;
            wheelFR.brakeTorque = 0f;
            wheelRL.brakeTorque = 0f;
            wheelRR.brakeTorque = 0f;

            brakeLights.SetActive(false);
            brakeLightsLed.SetActive(false);

            isBraking = false;
        }
    }

    /// <summary>
    /// If the car hits boost objects placed on the foot of uphill slopes, it gets a boost.
    /// As soon as the car reaches the end of the slope, it will hit an unboost object and
    /// remove the boost value.
    /// </summary>
    /// <param name="collider">Boost object with a certain tag</param>
    private void OnTriggerEnter(Collider boostObject)
    {
        // Used for climbing uphill
        if (boostObject.gameObject.tag == "Boost")
        {
            maxMotorTorque = maxBoostMotorTorque;
        }

        // Used after reaching the end of the slope
        if (boostObject.gameObject.tag == "Unboost")
        {
            maxMotorTorque = maxMotorTorqueCache;
        }
    }
}
