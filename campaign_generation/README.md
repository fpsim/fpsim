Per your request, I'm attaching a zip file that contains the some files for campaign generation.

* **CampaignClass.py** – Generated Python classes for configuring campaigns
* **CampaignEnum.py** – Generated Python enums that goes with the CampaignClass.py file.
* **Campaign-RC.json** – Example output of the GenerateCampaignFP.py file
* **Campaign-RCM.json** – Example output of the GenerateCampaginRCM.py file.
* **GenerateCampaignFP.py** – An good example of how to write an FP campaign that uses RandomChoice.  The campaign-RC.json file is an example of the output.  Originally, the campaign.json file in regression 2_FP_SimpleCampaign was the file generated from this script.  The file in the regression test has changed due to issues found in the campaign logic.
* **GenerateCampaignRCM.py** – This is an example that took the GenerateCampaignFP.py and hacked in some code to generate a CampaignEvent using RandomChoiceMatrix.  It is not meant to generate something useful but to show that you can generate a RandomChoiceMatrix object.

NOTE:  These scripts depend on DtkTools.

(Daniel Bridenbecker – 2019aug27)