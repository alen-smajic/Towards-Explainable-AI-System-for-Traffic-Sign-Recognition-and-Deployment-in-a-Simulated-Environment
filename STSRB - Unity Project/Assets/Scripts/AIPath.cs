using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// This class visualizes the path between neighbouring elements of the nodes list
/// inside the editor. Draws a line between the nodes.
/// Has to be attached to the parent gameobject which contains all the node gameobjects.
/// </summary>
public class AIPath : MonoBehaviour
{
    public Color lineColor;
    private List<Transform> nodes = new List<Transform>();

    /// <summary>
    /// Makes the path visible inside the editor.
    /// </summary>
    public void OnDrawGizmos()
    {
        // Set color for editor
        Gizmos.color = lineColor;

        // Get all nodes
        Transform[] pathTransforms = GetComponentsInChildren<Transform>();

        // Fills the nodes list with elements
        for(int i = 0; i < pathTransforms.Length; i++)
        {
            if(pathTransforms[i] != transform)
            {
                nodes.Add(pathTransforms[i]);
            }
        }

        // Draws lines between every two nodes and a sphere around every single node
        for (var i = 0; i < nodes.Count; i++)
        {
            var currentNode = nodes[i].position;
            var previousNode = Vector3.zero;

            if (i > 0)
            {
                previousNode = nodes[i - 1].position;
            }
            else if(i == 0 && nodes.Count > 1)
            {
                previousNode = nodes[nodes.Count - 1].position;
            }

            Gizmos.DrawLine(previousNode, currentNode); // Draw line between two nodes
            Gizmos.DrawWireSphere(currentNode, 0.3f); // Draw a sphere around every node
        }
    }
}