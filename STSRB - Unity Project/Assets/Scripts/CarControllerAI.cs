using System.Collections.Generic;
using System.Collections;
using UnityEngine;

/// <summary>
/// This class controlls and drives the car autonomously.
/// </summary>
public class CarControllerAI : MonoBehaviour
{
    [Header("AI Specfications")]
    public Transform drivingPath;
    public float criticalWaypointDistance;
    public float startBrakingDistance;
    public float maxCriticalSpeed;
    public float maxBoostCriticalSpeed;
    public float maxCriticalMotorTorque;
    public float maxBoostMotorTorque;
    public float waypointSensorSensitivity;

    [Header("Car Objects")]
    public WheelCollider wheelFL;
    public WheelCollider wheelFR;
    public WheelCollider wheelRL;
    public WheelCollider wheelRR;
    public GameObject waypointSensor;
    public GameObject brakeLights;
    public GameObject brakeLightsLed;
    public GameObject steeringWheel;

    [Header("Car Specifications")]
    public Vector3 centerOfMass;
    public float maxSteerAngle;
    public float maxMotorTorque;
    public float maxBreakTorque;
    public float maxSpeed;

    private List<Transform> nodes;
    private int currentNode = 0;
    private float currentSpeed = 0f;
    private float maxCriticalSpeedCache;
    private bool isBraking = false;
    private bool isBoost = false;
    private bool brakeLightsActive = false;

    // Start is called before the first frame update
    private void Start()
    {
        // Applies new center of mass to the car
        GetComponent<Rigidbody>().centerOfMass = centerOfMass;

        // Fills the nodes list with the waypoint coordinates from the drivingPath gameobject
        nodes = new List<Transform>();
        Transform[] pathTransforms = drivingPath.GetComponentsInChildren<Transform>();
        for (int i = 0; i < pathTransforms.Length; i++)
        {
            if (pathTransforms[i] != drivingPath.transform)
            {
                nodes.Add(pathTransforms[i]);
            }
        }

        currentNode = 0;
        currentSpeed = 0f;
        maxCriticalSpeedCache = maxCriticalSpeed;
        isBraking = false;
        isBoost = false;
        brakeLightsActive = false;
    }

    // Update is called once per frame
    private void FixedUpdate()
    {
        // Autonomous driving once the simulation is started
        if(UI.simulationActive)
        {
            CheckWaypointDistance();
            Drive();
            ApplySteer();
            Brake();
        }
    }

    /// <summary>
    /// Checks if the current target waypoint has been reached and assigns a new one as the target.
    /// </summary>
    private void CheckWaypointDistance()
    {
        // Uses the waypoint sensor for measuring the distance
        if (Vector3.Distance(waypointSensor.transform.position, 
            nodes[currentNode].position) < waypointSensorSensitivity)
        {
            if (currentNode == nodes.Count - 1)
            {
                // If final waypoint has been reached, continue with the first one
                currentNode = 0;
            }
            else
            {
                currentNode++;
            }
        }
    }

    /// <summary>
    /// Applies motor torque to all car wheels, while maintaining safety of driving in dangerous areas.
    /// </summary>
    private void Drive()
    {
        // Calculates the current speed of the car
        currentSpeed = 2 * Mathf.PI * wheelRL.radius * wheelRL.rpm * 60 / 1000;

        /// Check driving state
        // Checks if the following 2 nodes are critically near each other
        // Hints at hard driving area => braking will be needed
        if(Vector3.Distance(nodes[currentNode].position, 
            nodes[(currentNode+1) % nodes.Count].position) <= criticalWaypointDistance)
        {
            /// Potentially dangerous state
            // Checks if the car is critically near the next waypoint, which was classified as dangerous area
            if(Vector3.Distance(transform.position, nodes[currentNode].position) <= startBrakingDistance)
            {
                /// Dangerous state
                if (currentSpeed > maxCriticalSpeed)
                {
                    // Holds maxCriticalSpeed within the dangerous area
                    wheelFL.motorTorque = 0f;
                    wheelFR.motorTorque = 0f;
                    wheelRL.motorTorque = 0f;
                    wheelRR.motorTorque = 0f;
                    isBraking = true;
                }
                else if (isBoost)
                {
                    // Gives more power to the car (used for driving uphill)
                    wheelFL.motorTorque = maxBoostMotorTorque;
                    wheelFR.motorTorque = maxBoostMotorTorque;
                    wheelRL.motorTorque = maxBoostMotorTorque;
                    wheelRR.motorTorque = maxBoostMotorTorque;
                    isBraking = false;
                }
                else
                {
                    // Reduces the motor torque inside the dangerous area
                    wheelFL.motorTorque = maxCriticalMotorTorque;
                    wheelFR.motorTorque = maxCriticalMotorTorque;
                    wheelRL.motorTorque = maxCriticalMotorTorque;
                    wheelRR.motorTorque = maxCriticalMotorTorque;
                    isBraking = false;
                }
            }
            /// Safe driving state for now
            // Normal driving with full speed
            else
            {
                normalDriving(currentSpeed);
            }
        }
        /// Safe driving state
        // Normal driving with full speed
        else
        {
            normalDriving(currentSpeed);
        }
    }

    /// <summary>
    /// Applies maxMotorTorque to the wheels, if the full speed has not been reached yet.
    /// Used for normal driving in safe state.
    /// </summary>
    /// <param name="currentSpeed">Current speed of the car</param>
    private void normalDriving(float currentSpeed)
    {
        // Normal driving with full speed
        if (currentSpeed < maxSpeed)
        {
            wheelFL.motorTorque = maxMotorTorque;
            wheelFR.motorTorque = maxMotorTorque;
            wheelRL.motorTorque = maxMotorTorque;
            wheelRR.motorTorque = maxMotorTorque;
            isBraking = false;
        }
        // Holds maximum speed value
        else
        {
            wheelFL.motorTorque = 0f;
            wheelFR.motorTorque = 0f;
            wheelRL.motorTorque = 0f;
            wheelRR.motorTorque = 0f;
            isBraking = false;
        }
    }

    /// <summary>
    /// Applies the new steering angle to the steering wheel and tires corresponding to the loaction 
    /// of the target node.
    /// </summary>
    private void ApplySteer()
    {
        // Gets the angle to the next target node
        Vector3 relativeVector = transform.InverseTransformPoint(nodes[currentNode].position);
        float newSteer = (relativeVector.x / relativeVector.magnitude) * maxSteerAngle;

        // Rotates the car tires in the right direction
        wheelFL.steerAngle = newSteer;
        wheelFR.steerAngle = newSteer;

        // Rotates the steering wheel in the right direction
        Quaternion nextRotation = Quaternion.Lerp(steeringWheel.transform.localRotation,
            Quaternion.Euler(16.643f, 0f, -newSteer * 3), Time.deltaTime * 2f);
        steeringWheel.transform.localRotation = nextRotation;
    }

    /// <summary>
    /// Applies braking to all tires.
    /// </summary>
    private void Brake()
    {
        if (isBraking)
        {
            // Activates the brakes
            wheelRL.brakeTorque = maxBreakTorque;
            wheelRR.brakeTorque = maxBreakTorque;
            wheelFL.brakeTorque = maxBreakTorque;
            wheelFR.brakeTorque = maxBreakTorque;

            if (!brakeLightsActive && !isBoost)
            {
                // Activates brake lights
                StartCoroutine(activateBrakeLights());
            }
        }
        else
        {
            // Deactivates the brakes
            wheelRL.brakeTorque = 0f;
            wheelRR.brakeTorque = 0f;
            wheelFL.brakeTorque = 0f;
            wheelFR.brakeTorque = 0f;
        }
    }

    /// <summary>
    /// Activates the brake lights for 3 seconds and deactivates them afterwards.
    /// Used to reduce brake light flackering.
    /// </summary>
    /// <returns></returns>
    IEnumerator activateBrakeLights()
    {
        brakeLightsActive = true;
        brakeLights.SetActive(true);
        brakeLightsLed.SetActive(true);

        yield return new WaitForSeconds(3f);

        brakeLightsActive = false;
        brakeLights.SetActive(false);
        brakeLightsLed.SetActive(false);
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
        if(boostObject.gameObject.tag == "Boost")
        {
            isBoost = true;
            maxCriticalSpeed = maxBoostCriticalSpeed;
        }

        // Used after reaching the end of the slope
        if (boostObject.gameObject.tag == "Unboost")
        {
            isBoost = false;
            maxCriticalSpeed = maxCriticalSpeedCache;
        }
    }
}
