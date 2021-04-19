using UnityEngine;

/// <summary>
/// This class is used to copy the transformation and rotation of a target wheel collider.
/// It is used to mimic (to translate) the wheel collider physics to the real wheel objects.
/// Applied to tire, disk, rim etc. Script must be attached to these objects.
/// </summary>
public class CarWheel : MonoBehaviour
{
    public WheelCollider targetWheel; 

    private Vector3 wheelPosition = new Vector3();
    private Quaternion wheelRotation = new Quaternion();

    // Update is called once per frame
    void Update()
    {
        targetWheel.GetWorldPose(out wheelPosition, out wheelRotation);

        // Applies the transformation and rotation to the target gameobject
        transform.position = wheelPosition;
        transform.rotation = wheelRotation;
    }
}
